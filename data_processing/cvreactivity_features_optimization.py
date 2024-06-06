import numpy as np
import pandas as pd

from data_processing.dataframe_processing_functions import extract_df_date_range


def extract_sbp_new(df):
    df_bp = df[['subject', 'effectiveDateTime', 'code', 'encounter', 'component', 'resourceType']]

    # process the subject column as only the value of reference
    df_bp = pd.concat([df_bp.drop(['subject'], axis=1), df_bp['subject'].apply(pd.Series)], axis=1)
    # inserting column at appropriate place
    ref = df_bp['reference']
    df_bp.drop(labels=['reference'], axis=1, inplace=True)
    df_bp.insert(0, 'subject_reference', ref)
    df_bp['effectiveDateTime'] = df_bp['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df_bp['effectiveDateTime'] = pd.to_datetime(df_bp['effectiveDateTime'])
    df_bp['effectiveDateTime'] = df_bp['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))
    df_test = df_bp['code'].apply(pd.Series)
    df_bp = pd.concat([df_bp, df_test], axis=1)
    # drop the not required columns
    df_bp.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df_bp['coding'].apply(pd.Series)
    df_bp = pd.concat([df_bp, df_test], axis=1)
    # drop the not required columns
    df_bp.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df_bp[0].apply(pd.Series)
    df_bp = pd.concat([df_bp, df_test], axis=1)
    # drop the not required columns
    df_bp.drop(labels=[0, 'system', 'display'], axis=1, inplace=True)
    df_bp.rename(columns={'code': 'activity_code'}, inplace=True)
    df_test = df_bp['encounter'].apply(pd.Series)
    df_bp = pd.concat([df_bp, df_test], axis=1)
    df_bp.drop(labels=['encounter', 0], axis=1, inplace=True)
    df_bp.rename(columns={'reference': 'encounter_reference'}, inplace=True)
    df_bp['subject_reference'] = df_bp['subject_reference'].astype("str")
    df_bp['subject_reference'] = df_bp['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)
    df_bp['encounter_reference'] = df_bp['encounter_reference'].astype("str")
    df_bp['encounter_reference'] = df_bp['encounter_reference'].apply(
        lambda x: x.replace("Encounter/", "", 1) if x.startswith("Encounter/") else x)
    df_test = df_bp['component'].apply(pd.Series)
    df_bp = pd.concat([df_bp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_bp[0]
    imp_col2 = df_bp[1]
    df_bp.drop(labels=[0, 1], axis=1, inplace=True)
    df_bp.insert(0, 'component_1', imp_col2)
    df_bp.insert(0, 'component_0', imp_col1)
    df_sbp = df_bp.copy()
    #### Extracting SBP values
    # ======Extract values wrt component_0
    # =======================================================================================
    df_test = df_sbp['component_0'].apply(pd.Series)
    df_sbp = pd.concat([df_sbp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_sbp['code']
    imp_col2 = df_sbp['valueQuantity']
    df_sbp.drop(labels=['code', 'valueQuantity'], axis=1, inplace=True)
    df_sbp.insert(0, 'component_0_valueQuantity', imp_col2)
    df_sbp.insert(0, 'component_0_code', imp_col1)

    # =======================================================================================
    df_test = df_sbp['component_0_code'].apply(pd.Series)
    df_sbp = pd.concat([df_sbp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_sbp['coding']
    df_sbp.drop(labels=['coding'], axis=1, inplace=True)
    df_sbp.insert(0, 'component_0_code_coding', imp_col1)

    # =======================================================================================
    df_test = df_sbp['component_0_code_coding'].apply(pd.Series)
    df_sbp = pd.concat([df_sbp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_sbp[0]
    df_sbp.drop(labels=[0], axis=1, inplace=True)
    df_sbp.insert(0, 'component_0_code_coding_0', imp_col1)

    # =======================================================================================
    df_test = df_sbp['component_0_code_coding_0'].apply(pd.Series)
    df_sbp = pd.concat([df_sbp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_sbp['code']
    imp_col2 = df_sbp['display']
    df_sbp.drop(labels=['code'], axis=1, inplace=True)
    df_sbp.insert(0, 'component_0_code_coding_0_code', imp_col1)
    df_sbp.insert(0, 'component_0_code_coding_0_display', imp_col2)

    # =======================================================================================
    df_test = df_sbp['component_0_valueQuantity'].apply(pd.Series)
    df_sbp = pd.concat([df_sbp, df_test], axis=1)
    # inserting column at appropriate place
    imp_col1 = df_sbp['unit']
    imp_col2 = df_sbp['value']
    df_sbp.drop(labels=['unit', 'value'], axis=1, inplace=True)
    df_sbp.insert(0, 'component_0_valueQuantity_value', imp_col2)
    df_sbp.insert(0, 'component_0_code_unit', imp_col1)

    # =======================================================================================
    df_sbp = df_sbp[['subject_reference', 'effectiveDateTime', 'activity_code', 'encounter_reference',
                     'component_0_valueQuantity_value']]
    df_sbp.rename(columns={'component_0_valueQuantity_value': 'obs'}, inplace=True)
    return df_sbp


# getting imp columns for derived ehi
# cv reactivity, pulsepressure, affect
def extract_new_ehi(df, s_date, e_date):
    # extracting required features from json
    df = df.dropna(subset=['code'])
    df = df[['subject', 'effectiveDateTime', 'code', 'valueQuantity', 'encounter', 'resourceType']]
    df = pd.concat([df.drop(['subject'], axis=1), df['subject'].apply(pd.Series)], axis=1)

    # inserting column at appropriate place
    ref = df['reference']
    df.drop(labels=['reference'], axis=1, inplace=True)
    df.insert(0, 'subject_reference', ref)

    # convert datetime into meaningful values
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x[0:x.find('.')])
    df['effectiveDateTime'] = pd.to_datetime(df['effectiveDateTime'])
    df['effectiveDateTime'] = df['effectiveDateTime'].apply(lambda x: x.replace(microsecond=0, second=0))

    df['Date'] = df['effectiveDateTime'].dt.date
    df['Time'] = df['effectiveDateTime'].dt.time
    date_min = min(df['Date'])
    date_max = max(df['Date'])
    # extract code
    df_test = df['code'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['code', 'text'], axis=1, inplace=True)
    df_test = df['coding'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['coding'], axis=1, inplace=True)
    df_test = df[0].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    # drop the not required columns
    df.drop(labels=['system', 0], axis=1, inplace=True)
    df.rename(columns={'code': 'activity_code'}, inplace=True)
    # process the subject column as only the value of reference
    df = pd.concat([df, df['valueQuantity'].apply(pd.Series)], axis=1)
    # df = df.drop([0], axis = 1)
    # inserting column at appropriate place
    ref = df['value']
    df.insert(3, 'obs', ref)
    df.drop(labels=['value', 'valueQuantity', 'unit', 'system', 'code'], axis=1, inplace=True)

    # reformatting the subject reference
    df['subject_reference'] = df['subject_reference'].astype("str")
    df['subject_reference'] = df['subject_reference'].apply(
        lambda x: x.replace("Patient/", "", 1) if x.startswith("Patient/") else x)

    df = extract_df_date_range(df, s_date, e_date, date_min, date_max)
    df_test = df['encounter'].apply(pd.Series)
    df = pd.concat([df, df_test], axis=1)
    df.drop(labels=['encounter', 0], axis=1, inplace=True)
    df.rename(columns={'reference': 'encounter_reference'}, inplace=True)
    df['encounter_reference'] = df['encounter_reference'].astype("str")
    df['encounter_reference'] = df['encounter_reference'].apply(
        lambda x: x.replace("Encounter/", "", 1) if x.startswith("Encounter/") else x)
    return df


# CV reactivity calculation using multiple or single ehi
def reactivity_process_df(df, ehi_type, resting, post_act):
    df_post = df[df.activity_code == post_act]
    user_id = df_post['subject_reference'].unique()
    df_rest = df[df.activity_code == resting]
    df_rest = df_rest[df_rest.subject_reference.isin(user_id)]
    df_rest = df_rest[['subject_reference', 'effectiveDateTime', 'encounter_reference', 'obs']]
    df_rest.rename(columns={'obs': f'{ehi_type}_obs_rest'}, inplace=True)
    # sorted by subject and date in ascending order
    df_rest = df_rest.sort_values(['subject_reference', 'effectiveDateTime'], ascending=True)
    # group by user id and list the prq values
    df_rest_final = df_rest.groupby(['subject_reference', 'encounter_reference']).agg(lambda x: list(x))
    df_rest_final = df_rest_final.reset_index()
    df_rest_final.rename(columns={}, inplace=True)
    df_post = df_post[['subject_reference', 'effectiveDateTime', 'encounter_reference', 'obs']]
    df_post.rename(columns={'obs': f'{ehi_type}_obs_post'}, inplace=True)
    # sorted by subject and date in ascending order
    df_post = df_post.sort_values(['subject_reference', 'effectiveDateTime'], ascending=True)
    # group by user id and list the prq values
    df_post_final = df_post.groupby(['subject_reference', 'encounter_reference']).agg(lambda x: list(x))
    df_post_final = df_post_final.reset_index()
    df_rest_final = df_rest_final.drop(columns='effectiveDateTime')
    df_post_final = df_post_final.drop(columns='effectiveDateTime')

    df_final = pd.merge(df_rest_final, df_post_final, on=['subject_reference', 'encounter_reference'],
                        how='inner')
    df_final[f'count_{ehi_type}_obs_rest'] = df_final[f'{ehi_type}_obs_rest'].apply(lambda x: len(x))
    df_final[f'count_{ehi_type}_obs_post'] = df_final[f'{ehi_type}_obs_post'].apply(lambda x: len(x))

    condition_rest = f'count_{ehi_type}_obs_rest'
    condition_post = f'count_{ehi_type}_obs_post'
    df_final = df_final[((df_final[condition_rest] >= 3) & (df_final[condition_post] >= 3))]
    df_final = cv_reactivity(df_final, ehi_type)
    return df_final


def cv_reactivity(df_final, ehi_type):
    df_final[f'mean_{ehi_type}_obs_rest'] = df_final[f'{ehi_type}_obs_rest'].apply(lambda x: np.mean(x))
    df_final[f'change_{ehi_type}_obs_post_rest'] = df_final.apply(lambda x: change(x, ehi_type), axis=1)
    df_final[f'flag_change_{ehi_type}_obs_post_rest'] = df_final[f'change_{ehi_type}_obs_post_rest'].apply(
        update_list_with_flag)
    df_final = df_final[['subject_reference', 'encounter_reference', f'flag_change_{ehi_type}_obs_post_rest']]
    return df_final


# definitions for calculating reactivity
def change(df, ehi_type):
    ll = df[f'{ehi_type}_obs_post']
    rest = df[f'mean_{ehi_type}_obs_rest']
    ll0 = []
    for elt in ll:
        ll0.append(np.round(elt - rest, 2))
    return ll0


def update_list_with_flag(lst):
    if not lst:
        return lst  # Return empty list if the original list is empty

    percentile_75 = pd.Series(lst).quantile(0.75)
    updated_list = ['Exaggerated' if val > percentile_75 else 'Normal' for val in lst]
    return updated_list


# calculating the final decision column values
def fetch_existing_columns(df, flag_columns):
    existing_columns = [col for col in flag_columns if col in df.columns]
    return existing_columns


def compare_lists(lists):
    final_list = []
    for i in range(len(lists[0])):
        if any(lst[i] == 'Exaggerated' for lst in lists):
            final_list.append('Exaggerated')
        else:
            final_list.append('Normal')
    return final_list


def determine_final_value(lst):
    count_exaggerated = lst.count('Exaggerated')
    total_elements = len(lst)

    if count_exaggerated / total_elements > 0.50:
        return 'Exaggerated'
    else:
        return 'Normal'


# updating resource list
def ehi_input(lst):
    out_lst = []
    for var in lst:
        if var == 'prq':
            out_lst.append("flag_change_cadphr-prq_obs_post_rest")
        elif var == 'sbp':
            out_lst.append("flag_change_cadphr-bloodpressure_obs_post_rest")
        elif var == 'hr':
            out_lst.append("flag_change_cadphr-heartrate_obs_post_rest")
    return out_lst


# appending required df
def lst_of_df_cvreact(lst1, df1, df2, df3):
    lst = []
    for var in lst1:
        if var == 'hr':
            lst.append(df1)
        elif var == 'sbp':
            lst.append(df2)
        elif var == 'prq':
            lst.append(df3)
    return lst


# calculating final reactivity column
# noinspection PyUnboundLocalVariable
def final_cv_reactivity(lst_df, lst_col):
    if len(lst_df) == 3:
        df_test = pd.merge(lst_df[0], lst_df[1], on=['subject_reference', 'encounter_reference'], how='inner')
        df_final = pd.merge(df_test, lst_df[2], on=['subject_reference', 'encounter_reference'], how='inner')
    elif len(lst_df) == 2:
        df_final = pd.merge(lst_df[0], lst_df[1], on=['subject_reference', 'encounter_reference'], how='inner')
    elif len(lst_df) == 1:
        df_final = lst_df[0]
    existing_columns = [col for col in lst_col if col in df_final.columns]
    df_final['FinalList'] = df_final[existing_columns].apply(compare_lists, axis=1)
    df_final['conclusion'] = df_final['FinalList'].apply(determine_final_value)
    return df_final
