# Configurtions and constants for the project
N_CLUSTERS = 3
RANDOM_SEED = 42

# Column names from the dataset
TIMESTAMP = "Timestamp"
AGE = "Age"
HOURS_PER_DAY = "Hours per day"
FAV_GENRE = "Fav genre"
MOST_LISTENED_GENRE = "Most listened genre"
ALIGNMENT = "Alignment"
MENTAL_HEALTH_INDEX = "Mental health index"

# Frequency columns have this prefix format: "Frequency [Rock]"
FREQ_PREFIX = "Frequency ["
FREQ_SUFFIX = "]"


# Mental health columns
ANXIETY = "Anxiety"
DEPRESSION = "Depression"
INSOMNIA = "Insomnia"
OCD = "OCD"

# Ordinal encoding for frequency scale
MENTAL_HEALTH_COLS = ["Anxiety", "Depression", "Insomnia", "OCD"]

# Mappings
FREQ_MAPPING = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3,
}

TARGET_GENRE_GROUPS = [
    "Group: Rock, Metal, Pop",
    "Group: Classical, Jazz, Folk",
    "Group: HipHop, RB, Rap"
]



