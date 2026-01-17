from favourite_genre_to_mental_health import run_question_two
from genre_mental_correlation import run_question_one
from utilities import get_logger

if __name__ == "__main__":
    logger = get_logger()
    run_question_one(logger)
    run_question_two(logger)