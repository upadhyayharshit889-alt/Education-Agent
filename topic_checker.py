syllabus_topics = [
    "Python basics",
    "SQL queries and joins",
    "Statistics",
    "Machine Learning",
    "Power BI dashboards",
    "Data Visualization"
]

student_notes = [
    "Python basics",
    "SQL queries and joins",
    "Power BI dashboards"
]

missing_topics = []

for topic in syllabus_topics:
    if topic not in student_notes:
        missing_topics.append(topic)

print("===== MISSING SYLLABUS TOPICS =====\n")

for topic in missing_topics:
    print("-", topic) 