"""An owning admission's settled answer, never transcript reconstruction."""


def admission_answer(result, fallback):
    answer = result.get('final_response')
    return answer if isinstance(answer, str) else fallback
