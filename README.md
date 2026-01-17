Project objectives
Assumptions
Thesis
Results

Key stages thesis 1:
<hadar fill here>


Project structure:
./main.py - start point of the project
./genre_mental_correlation.py - implementation for our first thesis
./favourite_genre_to_mental_health.py - implementation for our second thesis
./utilities.py - Utilities file for common function
./visualize.py - Functions that help with visualization
./consts.py - Constants across the code
./tests
    test_q1.py - Tests for thesis number one
    test_q2_alignment.py - Tests for thesis number two

## Thesis 1


## Thesis 2
Is alignment between a participant's favorite genre and the genre they listen to most
associated with better mental health outcomes?

------------------------------------------------------------
Key stages:
Main pipeline (minimal): load the data → clean the data → encode the data → compute variables (genre, alignment, mental index) → test → plot (visualize) → interpret.

Operational definitions:
1) "Most listened genre":
   For each participant, we look at all columns that start with "Frequency [".
   We convert frequency categories to an ordinal scale:
   Never=0, Rarely=1, Sometimes=2, Very frequently=3
   Then, the genre with the highest score (row-wise) is chosen via idxmax().

2) "Alignment":
   Alignment = True if Fav genre == Most_Listened_Genre, otherwise False.

3) "Mental_Health_Index":
   Mental_Health_Index = mean(Anxiety, Depression, Insomnia, OCD)
   (Each variable is on a 0–10 scale, so the mean remains on a 0–10 scale.)

------------------------------------------------------------
Hypotheses:
H0 (null): mean Mental_Health_Index is the same in aligned and not-aligned participants.
H1 (alt) : mean Mental_Health_Index differs between aligned and not-aligned participants.

------------------------------------------------------------
Statistical test:
Independent samples t-test (Aligned vs Not aligned), alpha = 0.05

Notes about interpretation:
- If p < 0.05 → reject H0 (significant difference)
- If p >= 0.05 → fail to reject H0 (no significant difference)
---

## Data Description
| קטגוריה            | עמודות כלולות                                                                 | Data type        | Statistical scale        | תיאור |
|--------------------|----------------------------------------------------------------------------------|------------------|--------------------------|-------|
| דמוגרפיה וכללי     | Timestamp, Age, Streaming service, Permissions                                   | object / float64 | Nominal / Continuous     | פרטי המשיב, גיל ואישורי פרטיות. |
| הרגלי האזנה        | Hours per day, BPM                                                               | float64          | Continuous               | כמות שעות האזנה וקצב מוזיקה. |
| רקע מוזיקלי        | Fav genre, Instrumentalist, Composer, While working, Exploratory, Foreign languages | object           | Nominal / Dichotomous    | ז'אנר מועדף, האם מנגן/מלחין והרגלי חשיפה. |
| תדירות ז'אנרים     | Frequency [16 Genres: Classical, Rock, Pop, Metal, Jazz, etc.]                  | object           | Ordinal                  | רמת צריכה של 16 ז'אנרים שונים. |
| בריאות מנטלית      | Anxiety, Depression, Insomnia, OCD                                                | float64          | Discrete (0–10)          | דירוג עצמי של מדדי חוסן נפשי. |
| השפעת המוזיקה      | Music effects                                                                    | object           | Nominal                  | השפעה סובייקטיבית (משפר/מחמיר). |


## Instructions for running the project
`python ./main.py`