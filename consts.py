# Ordinal encoding for frequency scale
HEALTH_COLS = ["Anxiety", "Depression", "Insomnia", "OCD"]

FREQ_MAPPING = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3,
}

FREQ_PREFIX = "Frequency ["

# Column names from the dataset
TIMESTAMP = "Timestamp"
AGE = "Age"
HOURS_PER_DAY = "Hours per day"

FAV_GENRE = "Fav genre"

# Frequency columns have this prefix format: "Frequency [Rock]"
FREQ_PREFIX = "Frequency ["
FREQ_SUFFIX = "]"

# Mental health columns
ANXIETY = "Anxiety"
DEPRESSION = "Depression"
INSOMNIA = "Insomnia"
OCD = "OCD"

MENTAL_HEALTH_COLS = [ANXIETY, DEPRESSION, INSOMNIA, OCD]

# Derived / engineered feature names
MOST_LISTENED_GENRE = "Most listened genre"
ALIGNMENT = "Alignment"
MENTAL_HEALTH_INDEX = "Mental health index"

TARGET_GENRE_GROUPS = [
    "Group: Rock, Metal, Pop",
    "Group: Classical, Jazz, Folk",
    "Group: HipHop, RB, Rap"
]