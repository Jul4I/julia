from julia_log import logger

log = logger.get_logger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    log.debug("Calculating levenshtein_distance among '%s' and '%s'", s1, s2)
    m, n = len(s1), len(s2)

    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
    result = dp[m][n]
    log.debug("levenshtein_distance result: %d", result)
    return result
