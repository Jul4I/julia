import math

import torch
from julia_log.logger import get_logger
from torch import nn, optim

logger = get_logger(__name__)


# --- Positional Encoding ---
# Objetivo: Proveer a cada token de entrada una información sobre su posición en la secuencia.
# Esto es crucial porque los Transformers no tienen un mecanismo recurrente para captar el orden.
# d_model: Dimensión de los embeddings.
# dropout: Probabilidad de aplicar Dropout para evitar el sobreajuste.
# max_len: Longitud máxima de secuencias que se esperan procesar.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        # Dropout es una técnica de regularización que consiste en apagar
        # (poner a cero) de forma aleatoria algunas neuronas
        # durante el entrenamiento para evitar que el modelo se sobre-adapte
        # a los datos de entrenamiento.
        self.dropout = nn.Dropout(p=dropout)

        # Se crea un tensor para almacenar las codificaciones posicionales
        pe = torch.zeros(max_len, d_model)  # Inicializa un tensor de ceros (max_len, d_model)
        # 'position' contiene los índices de posición para cada token: (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # unsqueeze(1) añade una dimensión extra para que cada posición
        # esté en su propia fila, resultando en forma (max_len, 1)

        # 'div_term' calcula una escala que varía de forma exponencial
        # para aplicar a las funciones seno y coseno.
        # Esto permite que cada dimensión tenga una frecuencia distinta,
        # facilitando al modelo capturar relaciones de largo alcance.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Se aplican funciones trigonométricas para generar las codificaciones:
        # En posiciones pares se utiliza el seno...
        pe[:, 0::2] = torch.sin(position * div_term)
        # ...y en posiciones impares el coseno.
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Si d_model es impar, se ajusta el tamaño para que coincida la dimensión
            pe[:, 1::2] = torch.cos(position * div_term)[:, : pe[:, 1::2].shape[1]]

        # Se añade una dimensión extra al tensor de codificaciones para representar el batch.
        # Esto permite que al sumar las codificaciones a los embeddings, la operación se realice de forma broadcast.
        pe = pe.unsqueeze(0)  # La forma pasa de (max_len, d_model) a (1, max_len, d_model)
        # register_buffer registra 'pe' como parte del modelo, pero no es un parámetro a optimizar.
        self.register_buffer("pe", pe)

    # x: Tensor de embeddings de forma (batch_size, seq_len, d_model)
    # Se suma la codificación posicional a los embeddings para inyectar información de posición.
    def forward(self, x) -> nn.Dropout:  # noqa: ANN001
        # Se recorta 'pe' para ajustarse a la longitud de la secuencia de entrada y se suma a 'x'
        x = x + self.pe[:, : x.size(1)]  # type: ignore  # noqa: PGH003
        # Se aplica Dropout para evitar sobreajuste; durante el entrenamiento algunas activaciones se ponen a cero.
        return self.dropout(x)


# --- Modelo Transformer ---
# Objetivo: Definir un modelo completo de Transformer que incluya tanto el encoder como el decoder,
# además de las capas de embedding y la salida final para mapear a un vocabulario.
# src_vocab_size: Tamaño del vocabulario de origen (input).
# tgt_vocab_size: Tamaño del vocabulario de destino (output).
# d_model: Dimensión de los embeddings y de las capas internas.
# nhead: Número de cabezas en la atención multi-cabeza.
# num_encoder_layers: Número de capas en el encoder.
# num_decoder_layers: Número de capas en el decoder.
# dim_feedforward: Dimensión de la red feedforward interna.
# dropout: Probabilidad de dropout en diversas capas para regularización.
class TransformerModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Definición de las capas de embedding para el input (source) y output (target)
        # Las embeddings transforman los índices de tokens en vectores de dimensión d_model.
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Se añaden codificaciones posicionales para que el modelo tenga información sobre la posición de cada token.
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # --- Encoder ---
        # Se define una capa del encoder que utiliza atención multi-cabeza y una red feedforward.
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        # Se apilan 'num_encoder_layers' de estas capas para formar el encoder completo.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- Decoder ---
        # Similar al encoder, se define una capa del decoder.
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        # Se apilan 'num_decoder_layers' de estas capas para formar el decoder completo.
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Capa final que mapea la salida del decoder (dimensión d_model) a logits sobre el vocabulario de destino.
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None) -> nn.Linear:  # noqa: ANN001
        """
        src: Tensor de entrada (batch_size, src_seq_len)
        tgt: Tensor de destino para el decoder (batch_size, tgt_seq_len)
        src_mask, tgt_mask, memory_mask: Máscaras opcionales para controlar la atención.
        """  # noqa: D205
        # --- Procesamiento de la secuencia de origen (encoder) ---
        # Se aplica la capa de embedding y se escala el resultado.
        # La escala (sqrt(d_model)) ayuda a estabilizar el entrenamiento.
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        # Se añade la codificación posicional a los embeddings para
        # inyectar la información de posición.
        src_emb = self.pos_encoder(src_emb)
        # Los módulos Transformer de PyTorch esperan la entrada con la
        # dimensión de secuencia en primer lugar: (seq_len, batch_size, d_model).
        src_emb = src_emb.transpose(0, 1)
        # Se procesa la secuencia a través del encoder para obtener
        # una representación intermedia (memory).
        memory = self.encoder(src_emb, mask=src_mask)

        # --- Procesamiento de la secuencia de destino (decoder) ---
        # Se aplica la capa de embedding para el target y se escala.
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # Se añade la codificación posicional a la secuencia destino.
        tgt_emb = self.pos_decoder(tgt_emb)
        # Se transpone para adecuar la forma (seq_len, batch_size, d_model).
        tgt_emb = tgt_emb.transpose(0, 1)
        # El decoder procesa la secuencia destino junto con la representación del encoder ('memory'),
        # aplicando una máscara para que cada posición solo vea tokens anteriores o actuales.
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # Se transpone la salida para volver a la forma (batch_size, seq_len, d_model).
        output = output.transpose(0, 1)
        # Se aplica la capa final lineal para obtener logits sobre el vocabulario de destino.
        return self.fc_out(output)


# --- Función para generar la máscara del decoder ---
# Genera una máscara triangular superior para el decoder, que evita que el modelo acceda a tokens futuros.
# Esto es esencial durante la generación de secuencias para evitar 'mirar hacia adelante'.
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    # Crea una matriz de unos de forma (sz, sz) y extrae la parte triangular superior.
    mask = torch.triu(torch.ones(sz, sz)) == 1
    # Convierte la máscara a tipo float y reemplaza los ceros con -infinito (-inf) para bloquear la atención
    # y los unos con 0.0, indicando que no hay restricción en esa posición.
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)


# --- Ejemplo genérico de entrenamiento ---
# Esta función ilustra un ciclo de entrenamiento básico utilizando el Transformer definido.
# Se generan datos sintéticos (aleatorios) para simular entradas y salidas.
def train_transformer() -> None:
    src_vocab_size = 1000  # Tamaño del vocabulario para la secuencia de origen
    tgt_vocab_size = 1000  # Tamaño del vocabulario para la secuencia de destino
    d_model = 512  # Dimensión de los embeddings y de las capas internas
    batch_size = 32  # Número de muestras que se procesan en paralelo
    src_seq_len = 20  # Longitud de la secuencia de origen
    tgt_seq_len = 20  # Longitud de la secuencia de destino
    num_epochs = 5  # Número de veces que se procesará todo el conjunto de datos sintético

    # Selección del dispositivo: se utiliza GPU si está disponible, en caso contrario CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciamos el modelo y lo movemos al dispositivo seleccionado.
    model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model=d_model).to(device)
    # Se define el optimizador Adam, que ajusta los parámetros del modelo minimizando la pérdida.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Se utiliza CrossEntropyLoss, adecuada para problemas de clasificación, ignorando el token de padding (índice 0).
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Se pone el modelo en modo entrenamiento, activando mecanismos como Dropout.
    model.train()
    for epoch in range(num_epochs):
        # --- Generación de datos sintéticos ---
        # Se generan tensores aleatorios que simulan secuencias de tokens.
        # Se evitan los ceros (que se usan para padding).
        src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len)).to(device)
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)

        # Para entrenar el decoder se utiliza teacher forcing:
        # Se utiliza la secuencia destino sin el último token como entrada.
        tgt_input = tgt[:, :-1]
        # La salida esperada es la secuencia destino sin el primer token.
        tgt_expected = tgt[:, 1:]

        # Se genera una máscara para el decoder que impide que cada posición acceda a tokens futuros.
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        optimizer.zero_grad()  # Se reinician los gradientes para evitar acumulación.
        # Se realiza la pasada forward por el modelo: se procesan la secuencia de origen y la entrada del decoder.
        output = model(src, tgt_input, tgt_mask=tgt_mask)
        # La salida tiene forma (batch_size, seq_len, tgt_vocab_size) y se reestructura para calcular la pérdida.
        output = output.reshape(-1, tgt_vocab_size)
        tgt_expected = tgt_expected.reshape(-1)

        # Se calcula la pérdida comparando la salida del modelo con la secuencia destino esperada.
        loss = criterion(output, tgt_expected)
        loss.backward()  # Se calcula el gradiente mediante backpropagation.
        optimizer.step()  # Se actualizan los parámetros del modelo usando el optimizador.

        # Se imprime el valor de la pérdida para monitorear el progreso del entrenamiento.
        msg = f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}"
        logger.info(msg)


# Si este script se ejecuta directamente, se inicia el proceso de entrenamiento.
if __name__ == "__main__":
    train_transformer()
