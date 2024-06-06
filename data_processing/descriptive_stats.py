import warnings

warnings.filterwarnings("ignore")


# noinspection PyUnusedLocal
def assessment_unique_description(df, rule_name):
    assessment_column = 'assessment'
    s = df[assessment_column].explode()
    list_val = s.dropna().tolist()
    total_values = len(list_val)  # Total count of values in list_val
    unique_values = set(list_val)
    unique_counts = {}
    for val in unique_values:
        unique_counts[val] = list_val.count(val)

    mode_value = s.mode().iloc[0]
    descriptive_stats = s.describe().round(2)

    descriptive_stats_str = (f"<br><b>{rule_name} Assessment Rules (unique) Statistics:</b><br>"
                             f"The dataset contains {total_values} unique users.<br>"
                             "<br><b>Counts of Unique Assessment Rules Fired:</b><br>")

    # Calculate percentage for each unique value
    percentages = {val: count / descriptive_stats['count'] * 100 for val, count in unique_counts.items()}
    for val, percent in percentages.items():
        descriptive_stats_str += f"The percentage of - {val.split(":")[0]} is {percent:.2f}%<br>"

    descriptive_stats_str += (f"<br>"
                              "Mode in Rules list is: {}<br>").format(mode_value.split(":")[0])

    return descriptive_stats_str


# noinspection PyUnusedLocal
def assessment_grouped_description(df, rule_name):
    assessment_column = 'assessment'
    s = df[assessment_column].explode()
    list_val = s.dropna().tolist()
    total_values = len(list_val)  # Total count of values in list_val
    unique_counts = {}

    # Count occurrences of strings starting with 'RF', 'PF', or 'ICF'
    rf_count = 0
    pf_count = 0
    icf_count = 0
    for val in list_val:
        if val.startswith('RF'):
            rf_count += 1
        elif val.startswith('PF'):
            pf_count += 1
        elif val.startswith('ICF'):
            icf_count += 1

    unique_counts['RF'] = rf_count
    unique_counts['PF'] = pf_count
    unique_counts['ICF'] = icf_count

    mode_value = s.mode().iloc[0]
    descriptive_stats = s.describe().round(2)

    descriptive_stats_str = (f"<br><b>{rule_name} Assessment Rules (grouped) Statistics:</b><br>"
                             f"The dataset contains {total_values} unique users.<br>"
                             "<br><b>Counts of Grouped Assessment Rules Fired:</b><br>")

    # Calculate percentage for each unique value
    percentages = {val: count / descriptive_stats['count'] * 100 for val, count in unique_counts.items()}
    for val, percent in percentages.items():
        descriptive_stats_str += f"The percentage of - {val.split(":")[0]} is {percent:.2f}%<br>"

    descriptive_stats_str += (f"<br>"
                              "Mode in Rules list is: {}<br>").format(mode_value.split(":")[0])

    return descriptive_stats_str
