from cod2 import run_question_two
from cod1 import run_question_one
from utilities import get_logger

if __name__ == "__main__":
    logger = get_logger()
    run_question_one(logger)
    run_question_two(logger)
