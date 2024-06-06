import os
from textwrap import wrap
import warnings
import matplotlib
import matplotlib.pyplot as plt
from data_processing import return_required_dataframes as rd

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
def assessment_rule_unique_criteria_bar(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    s = df['assessment'].explode()
    s = s[s != ""]
    # Count the frequency of each rule
    assessment_counts = s.value_counts()

    # Calculate percentages
    total_assessments = len(s)
    percentages = (assessment_counts / total_assessments) * 100

    # # Define colors based on assessment rule prefixes and suffixes
    # colors = ['tomato' if label.startswith('RF') else 'green' if label.startswith('PF') else '#808000' if label.startswith('ICF') else None for label in assessment_counts.index]

    colors=[]
    # Extract colors from suffixes and update the colors list
    for i, label in enumerate(assessment_counts.index):
        prefix, color = label.split(":")
        colors.append(color)
    # Extract prefixes from assessment rule labels
    prefixes = [label.split(":")[0] for label in assessment_counts.index]

    # Plotting
    plt.bar(prefixes, percentages, color=colors, edgecolor='black')

    # Rotate x-axis labels vertically
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better fit
    # Set labels and title
    plt.xlabel('Assessment Rule')
    plt.ylabel('User Percentage')
    title = f'Bar Graph for {rule_name} (Unique Assessment Rules)'
    if start_age:
        title += f' with age group ({start_age}-{end_age})' if end_age else f' with age group >= {start_age}'
    elif end_age:
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender ({data_cond})'

    if start_d and end_d:
        title += f' during {start_d} to {end_d}'
    elif start_d:
        title += f' from {start_d} onwards'
    elif end_d:
        title += f' up to {end_d}'
    else:
        title += ' for all dates'

    plt.title("\n".join(wrap(title)))
    plt.tight_layout()
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    # Save the plot
    filename = f"bar_graph_unique_factors_assessment.png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}') # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    plt.savefig(file_path)
    plt.close()
    return filename

def assessment_rule_grouped_criteria_bar(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    s = df['assessment'].explode()
    # Count occurrences of RF, PF, and ICF strings
    rf_count = s.str.startswith('RF').sum()
    pf_count = s.str.startswith('PF').sum()
    icf_count = s.str.startswith('ICF').sum()

    # Calculate percentages
    total_assessments = len(s)
    rf_percentage = (rf_count / total_assessments) * 100
    pf_percentage = (pf_count / total_assessments) * 100
    icf_percentage = (icf_count / total_assessments) * 100

    # Plot histogram
    categories = ['RF', 'PF', 'ICF']
    counts = [rf_percentage, pf_percentage, icf_percentage]
    colors = ['tomato', 'green', 'lightgrey']

    plt.bar(categories, counts, color=colors)
    plt.xlabel('Assessment Rule Category')
    plt.ylabel('User Percentage')
    title = f'Bar Graph for {rule_name} (Grouped Assessment Rules)'
    if start_age:
        title += f' with age group ({start_age}-{end_age})' if end_age else f' with age group >= {start_age}'
    elif end_age:
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender ({data_cond})'

    if start_d or end_d:
        title += f' during {start_d} to {end_d}' if start_d and end_d else f' from {start_d} onwards' if start_d else f' up to {end_d}'
    else:
        title += ' for all dates'
    plt.title("\n".join(wrap(title)))
    plt.tight_layout()
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    # Save the plot
    filename = f"bar_graph_grouped_factors_{'assessment'}.png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')  # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return filename

def assessment_rule_KF_criteria_bar(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    KF_column = 'KF_flag'
    s = df[KF_column].explode()

    # Count the frequency of KF and non-KF
    count_0 = (s == 0).sum()
    count_1 = (s == 1).sum()
    total_assessments = len(s)

    # Calculate percentages
    percentage_0 = (count_0 / total_assessments) * 100
    percentage_1 = (count_1 / total_assessments) * 100

    # Plot histogram
    categories = ['KF', 'Non_KF']
    percentages = [percentage_1, percentage_0]
    colors = ['powderblue', 'deepskyblue']

    plt.bar(categories, percentages, color=colors)
    plt.xlabel('Assessment Rule Category')
    plt.ylabel('User Percentage')
    title = f'Bar Graph for {rule_name} (KF and Non KF-Assessment Rules)'
    if start_age:
        title += f' with age group ({start_age}-{end_age})' if end_age else f' with age group >= {start_age}'
    elif end_age:
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender ({data_cond})'

    if start_d or end_d:
        title += f' during {start_d} to {end_d}' if start_d and end_d else f' from {start_d} onwards' if start_d else f' up to {end_d}'
    else:
        title += ' for all dates'

    plt.title("\n".join(wrap(title)))
    plt.tight_layout()  # Ensure proper layout
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    # Save the plot
    filename = f"bar_graph_{KF_column}.png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')  # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return filename

def assessment_rule_unique_KF_criteria_bar(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    KF_column = 'KF_flag'
    assessment_column = 'assessment'

    # Filter the dataframe for rows where KF_flag is 1
    filtered_df = df[df[KF_column] == 1]

    if filtered_df.empty:
        print(f"No data with KF_flag = 1 for rule {rule_name}.")
        return ""

    # Get counts of each unique assessment value
    assessment_counts = filtered_df[assessment_column].value_counts()

    # Calculate percentage for each assessment value
    total_assessments = assessment_counts.sum()
    assessment_percentages = (assessment_counts / total_assessments) * 100
    prefixes = [label.split(":")[0] for label in assessment_counts.index]
    # Plot bar graph with percentage
    plt.bar(prefixes, assessment_percentages.values, color=['powderblue', 'deepskyblue', 'steelblue', 'skyblue'])
    plt.xlabel('Assessment Type')
    plt.ylabel('User Percentage')
    title = f'Bar Graph for {rule_name} (Unique KF-Assessment Rules)'
    if start_age:
        title += f' with age group ({start_age}-{end_age})' if end_age else f' with age group >= {start_age}'
    elif end_age:
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender ({data_cond})'

    if start_d or end_d:
        title += f' during {start_d} to {end_d}' if start_d and end_d else f' from {start_d} onwards' if start_d else f' up to {end_d}'
    else:
        title += ' for all dates'

    plt.xticks(rotation=45, ha='right')  # Rotate labels for better fit
    plt.title("\n".join(wrap(title)))
    plt.tight_layout()  # Ensure proper layout

    # Adjust figure size if needed
    plt.gcf().set_size_inches(10, 6)

    # Save the plot
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')
    os.makedirs(static_folder, exist_ok=True)  # Ensure directory exists
    filename = f"bar_graph_unique_{KF_column}-{rule_name}.png"
    file_path = os.path.join(static_folder, filename)
    plt.savefig(file_path)
    plt.close()
    return filename


def assessment_rule_unique_criteria_pie_chart(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    s = df['assessment'].explode()
    s = s[s != ""]
    assessment_counts = s.value_counts().sort_index(ascending=False)

    total_values = len(s)
    percentages = assessment_counts / total_values * 100

    colors = []

    # Extract colors from suffixes and update the colors list
    for i, label in enumerate(assessment_counts.index):
        prefix, color = label.split(":")
        colors.append(color)

    assessment_counts = assessment_counts[assessment_counts != 0]
    # Check if any percentage is less than 5%
    explode = [0.08] * len(assessment_counts)
    for percentage in percentages:
        if percentage < 5:
            explode = [0.15] * len(assessment_counts)
    prefixes = [label.split(":")[0] for label in assessment_counts.index]
    plt.pie(assessment_counts, labels=prefixes, autopct='%1.1f%%', startangle=150, explode=explode, colors=colors)
    title = f'Pie Chart for {rule_name}(Unique Assessment Rules)'
    if start_age != "" and end_age != "":
        title += f' with age group ({start_age}-{end_age})'
    elif start_age != "":
        title += f' with age group >= {start_age}'
    elif end_age != "":
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender({data_cond})'

    if start_d != "" and end_d != "":
        title += f' during {start_d} to {end_d}'
    elif start_d != "":
        title += f' from {start_d} onwards'
    elif end_d != "":
        title += f' up to {end_d}'
    else:
        title += ' for all dates'
    plt.title("\n".join(wrap(title)))
    plt.tight_layout()  # Ensure proper layout
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    filename = "pie_chart_unique_factors_assessment.png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')  # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return filename


def assessment_rule_grouped_criteria_pie_chart(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    s = df['assessment'].explode()

    # Count occurrences of RF, PF, and ICF strings
    rf_count = s.str.startswith('RF').sum()
    pf_count = s.str.startswith('PF').sum()
    icf_count = s.str.startswith('ICF').sum()
    assessment_counts = [rf_count, pf_count, icf_count]
    assessment_labels = ['RF', 'PF', 'ICF']

    # Define specific colors for each assessment label
    color_map = {'RF': 'tomato', 'PF': 'green', 'ICF': 'lightgrey'}

    # Filter out labels with count 0 and map colors
    assessment_counts, assessment_labels = zip(
        *[(count, label) for count, label in zip(assessment_counts, assessment_labels) if count != 0])
    colors = [color_map[label] for label in assessment_labels]

    # Create pie chart
    explode = [0.05] * len(assessment_counts)  # Default explode value
    for count in assessment_counts:
        if count / sum(assessment_counts) * 100 < 5:
            explode = [0.15] * len(assessment_counts)  # Set explode to 0.15 if any count is less than 5%
            break
    plt.pie(assessment_counts, explode=explode, labels=assessment_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Title and formatting
    title = f'Pie Chart for {rule_name}(Grouped Assessment Rules)'
    if start_age != "" and end_age != "":
        title += f' with age group ({start_age}-{end_age})'
    elif start_age != "":
        title += f' with age group >= {start_age}'
    elif end_age != "":
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender({data_cond})'

    if start_d != "" and end_d != "":
        title += f' during {start_d} to {end_d}'
    elif start_d != "":
        title += f' from {start_d} onwards'
    elif end_d != "":
        title += f' up to {end_d}'
    else:
        title += ' for all dates'

    plt.title("\n".join(wrap(title)))
    # Save and close plot
    filename = "pie_chart_grouped_factors_assessment.png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')  # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(file_path)
    plt.close()
    return filename


def assessment_rule_KF_criteria_pie_chart(df, rule_name, start_age, end_age, start_d, end_d, data_cond):
    KF_column = 'KF_flag'
    s = df[KF_column].explode()
    # Get the count of 0s and 1s in the Series s
    count_0 = (s == 0).sum()
    count_1 = (s == 1).sum()

    # Plot pie chart
    categories = ['KF', 'Non_KF']
    counts = [count_1, count_0]
    explode = (0.25, 0)  # Explode the first slice
    plt.pie(counts, labels=categories, explode=explode, autopct='%1.1f%%', colors=['powderblue', 'deepskyblue'])
    plt.axis('equal')
    # Title and formatting
    title = f'Pie Chart for {rule_name}(KF and Non KF Rules)'
    if start_age != "" and end_age != "":
        title += f' with age group ({start_age}-{end_age})'
    elif start_age != "":
        title += f' with age group >= {start_age}'
    elif end_age != "":
        title += f' with age group <= {end_age}'
    else:
        title += ' for all age groups'

    if data_cond in ['all', 'male', 'female']:
        title += f' for gender({data_cond})'

    if start_d != "" and end_d != "":
        title += f' during {start_d} to {end_d}'
    elif start_d != "":
        title += f' from {start_d} onwards'
    elif end_d != "":
        title += f' up to {end_d}'
    else:
        title += ' for all dates'

    plt.title("\n".join(wrap(title)))
    plt.tight_layout()
    # Adjust figure size if needed (Set figure size to 10x6 inches)
    plt.gcf().set_size_inches(10, 6)
    filename = "pie_chart_" + KF_column + ".png"
    static_folder = os.path.join(os.getcwd(), f'output/{rule_name}')  # Assumes 'static' is in the current working directory
    file_path = os.path.join(static_folder, filename)
    plt.savefig(file_path)
    plt.close()
    return filename
