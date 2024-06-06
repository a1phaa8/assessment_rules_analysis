import json
import os

from data_processing.activity_code import *
from data_processing.cardiac_index import cardiac_index
from data_processing.cvreactivity_features_optimization import *
from data_processing.dataframe_processing_functions import *

json_folder = "input/json_folder"


def process_common_df(rule_names, df_patient, common_params, trend_window, gender, s_age, e_age):
    sliding_window = common_params[2]
    min_date = ""
    max_date = ""
    processed_dataframes = {}
    bp_flag, hr_flag, hrv_flag, prq_flag, cad_flag = 0, 0, 0, 0, 0

    for rule_name in rule_names:
        if rule_name == "":
            print("NULL\n")
            pass
        if rule_name in ['ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule', 'ruleset4_V3_1_extreme_SBP_rule',
                         'ruleset4_V3_1_extreme_DBP_rule', 'ruleset5_V3_1_progression_sbp_rule',
                         'ruleset5_V3_1_progression_dbp_rule',
                         'ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule']:
            if bp_flag == 0:
                sbp_code = activity_code('resting', 'cadphr-sbp')
                dbp_code = activity_code('resting', 'cadphr-dbp')
                bp_json_file_path = os.path.join(json_folder, "cadphr-bloodpressure.json")
                with open(bp_json_file_path, 'r') as bp_json_file:
                    bp_data = json.load(bp_json_file)
                    bp_dataframe = pd.DataFrame(bp_data)

                processed_sbp_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    bp_dataframe,
                    sbp_code,
                    df_patient,
                    *common_params,
                    'cadphr-sbp'
                )

                processed_dbp_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    bp_dataframe,
                    dbp_code,
                    df_patient,
                    *common_params,
                    'cadphr-dbp'
                )
                processed_dataframes["sbp_df"] = [processed_sbp_df, min_age, max_age, min_bmi, max_bmi]
                processed_dataframes["dbp_df"] = [processed_dbp_df, min_age, max_age, min_bmi, max_bmi]
                bp_flag = 1
            if rule_name in ['ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule']:
                # Merge SBP and DBP DataFrames
                merged_dataframe = pd.merge(processed_sbp_df[
                                                ['subject_reference', 'gender', 'age', 'height', 'weight', 'BMI', 'SBP',
                                                 'cadphr-sbp_grade']],
                                            processed_dbp_df[['subject_reference', 'DBP', 'cadphr-dbp_grade']],
                                            on=['subject_reference'], how='inner')
                # Calculate 'Pulse Pressure'
                merged_dataframe['Pulse Pressure'] = merged_dataframe['SBP'] - merged_dataframe['DBP']
                processed_pulsepressure_dataframe = process(
                    merged_dataframe,
                    "",
                    df_patient,
                    *common_params,
                    'cadphr-pulsepressure'
                )
                processed_dataframes["pp_df"] = [processed_pulsepressure_dataframe, min_age, max_age, min_bmi, max_bmi]
            if rule_name in ['ruleset5_V3_1_progression_sbp_rule']:
                df_sbp_prog = processed_dataframes['sbp_df'][0]
                std_deviation = 'cadphr-sbp_std'
                df_sbp_prog[std_deviation] = df_sbp_prog['obs'].apply(
                    lambda x: calculate_std_with_sliding_window(x, sliding_window))
                df_sbp_prog['num_std_samples'] = df_sbp_prog[std_deviation].apply(lambda x: len(x))
                df_sbp_prog = df_sbp_prog[df_sbp_prog.num_std_samples >= 10]
                df_sbp_prog = df_sbp_prog.apply(
                    lambda x: extract_format_find_trend(x, trend_window), axis=1)
                processed_dataframes['sbp_progression_df'] = [df_sbp_prog, min_age, max_age, min_bmi, max_bmi]
            if rule_name in ['ruleset5_V3_1_progression_dbp_rule']:
                df_dbp_prog = processed_dataframes['dbp_df'][0]
                std_deviation = 'cadphr-dbp_std'
                df_dbp_prog[std_deviation] = df_dbp_prog['obs'].apply(
                    lambda x: calculate_std_with_sliding_window(x, sliding_window))
                df_dbp_prog['num_std_samples'] = df_dbp_prog[std_deviation].apply(lambda x: len(x))
                df_dbp_prog = df_dbp_prog[df_dbp_prog.num_std_samples >= 10]
                df_dbp_prog = df_dbp_prog.apply(
                    lambda x: extract_format_find_trend(x, trend_window), axis=1)
                processed_dataframes['dbp_progression_df'] = [df_dbp_prog, min_age, max_age, min_bmi, max_bmi]
            if rule_name in ['ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule']:
                merged_sbp_dbp = pd.merge(
                    processed_sbp_df[['subject_reference', 'SBP', 'cadphr-sbp_grade', 'age', 'gender']],
                    processed_dbp_df[['subject_reference', 'DBP', 'cadphr-dbp_grade']],
                    on='subject_reference', how='inner')
                processed_dataframes['sbp_dbp_merged'] = [merged_sbp_dbp, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset6_V3_1_combinatorial_hr_hrv_rule',
                         'ruleset4_V2_1_extreme_Heart_Rate_rule', 'ruleset6_V4_combinatorial_rules_target_HR_rule',
                         'ruleset5_V3_1_progression_rhr_rule']:
            if rule_name in ['ruleset6_V4_combinatorial_rules_target_HR_rule']:
                hr_code = activity_code('resting', 'cadphr-heartrate')
                hr_json_file_path = os.path.join(json_folder, "cadphr-heartrate.json")
                with open(hr_json_file_path, 'r') as json_file:
                    hr_data = json.load(json_file)
                    hr_dataframe = pd.DataFrame(hr_data)
                processed_target_hr_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    hr_dataframe,
                    hr_code,
                    df_patient,
                    *common_params,
                    'cadphr-targetHR'
                )
                processed_dataframes['targetHR_df'] = [processed_target_hr_dataframe, min_age, max_age, min_bmi,
                                                       max_bmi]
            if hr_flag == 0:
                hr_code = activity_code('resting', 'cadphr-heartrate')
                hr_json_file_path = os.path.join(json_folder, "cadphr-heartrate.json")
                with open(hr_json_file_path, 'r') as json_file:
                    hr_data = json.load(json_file)
                    hr_dataframe = pd.DataFrame(hr_data)
                processed_hr_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    hr_dataframe,
                    hr_code,
                    df_patient,
                    *common_params,
                    'cadphr-heartrate'
                )
                processed_hr_df.rename(columns={'Latest': 'Latest_HR_Value'}, inplace=True)
                processed_dataframes["heartrate_df"] = [processed_hr_df, min_age, max_age, min_bmi, max_bmi]
                hr_flag = 1
            if rule_name in ['ruleset5_V3_1_progression_rhr_rule']:
                df_heartrate = processed_dataframes['heartrate_df'][0]
                std_deviation = 'cadphr-heartrate_std'
                df_heartrate[std_deviation] = df_heartrate['obs'].apply(
                    lambda x: calculate_std_with_sliding_window(x, sliding_window))
                df_heartrate['num_std_samples'] = df_heartrate[std_deviation].apply(lambda x: len(x))
                df_heartrate = df_heartrate[df_heartrate.num_std_samples >= 10]
                df_heartrate = df_heartrate.apply(
                    lambda x: extract_format_find_trend(x, trend_window), axis=1)
                processed_dataframes['rhr_progression_df'] = [df_heartrate, min_age, max_age, min_bmi, max_bmi]

        if rule_name in ['ruleset6_V3_1_combinatorial_hr_hrv_rule', 'ruleset4_V2_1_extreme_Heart_Rate_Variability_rule',
                         'ruleset5_V3_1_progression_hrv_rule']:
            if hrv_flag == 0:
                hrv_code = activity_code('resting', 'cadphr-hrv')
                hrv_json_file_path = os.path.join(json_folder, "cadphr-hrv.json")
                with open(hrv_json_file_path, 'r') as json_file:
                    hrv_data = json.load(json_file)
                    hrv_dataframe = pd.DataFrame(hrv_data)
                processed_hrv_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    hrv_dataframe,
                    hrv_code,
                    df_patient,
                    *common_params,
                    'cadphr-hrv'
                )
                processed_hrv_df.rename(columns={'Latest': 'Latest_HRV_Value'}, inplace=True)
                processed_dataframes["hrv_df"] = [processed_hrv_df, min_age, max_age, min_bmi, max_bmi]
                hrv_flag = 1
            if rule_name in ['ruleset5_V3_1_progression_hrv_rule']:
                df_hrv = processed_dataframes['hrv_df'][0]
                std_deviation = 'cadphr-hrv_std'
                df_hrv[std_deviation] = df_hrv['obs'].apply(
                    lambda x: calculate_std_with_sliding_window(x, sliding_window))
                df_hrv['num_std_samples'] = df_hrv[std_deviation].apply(lambda x: len(x))
                df_hrv = df_hrv[df_hrv.num_std_samples >= 10]
                df_hrv = df_hrv.apply(
                    lambda x: extract_format_find_trend(x, trend_window), axis=1)
                processed_dataframes['hrv_progression_df'] = [df_hrv, min_age, max_age, min_bmi, max_bmi]

        # condition to check for hr and hrv combinatorial rule
        # and merge both dataframes to store as one dataframe
        if rule_name in ['ruleset6_V3_1_combinatorial_hr_hrv_rule']:
            merged_hr_hrv = pd.merge(
                processed_dataframes['heartrate_df'][0][
                    ['subject_reference', 'cadphr-heartrate_grade', 'Latest_HR_Value', 'age', 'gender']],
                processed_dataframes['hrv_df'][0][
                    ['subject_reference', 'cadphr-hrv_grade', 'Latest_HRV_Value']],
                on='subject_reference', how='inner')
            processed_dataframes['hr_hrv_merged'] = [merged_hr_hrv, min_age, max_age, min_bmi, max_bmi]

        if rule_name in ['ruleset4_V2_1_extreme_PRQ_rule',
                         'ruleset5_V4_progression_prq_rule']:
            if prq_flag == 0:
                prq_code = activity_code('resting', 'cadphr-prq')
                prq_json_file_path = os.path.join(json_folder, "cadphr-prq.json")
                with open(prq_json_file_path, 'r') as json_file:
                    prq_data = json.load(json_file)
                    prq_dataframe = pd.DataFrame(prq_data)
                processed_prq_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    prq_dataframe,
                    prq_code,
                    df_patient,
                    *common_params,
                    "cadphr-prq"
                )
                processed_prq_df.rename(columns={'Latest': 'Latest_PRQ_Value'}, inplace=True)
                processed_dataframes["prq_df"] = [processed_prq_df, min_age, max_age, min_bmi, max_bmi]
                prq_flag = 1
            if rule_name in ['ruleset5_V4_progression_prq_rule']:
                df_prq = processed_dataframes['prq_df'][0]
                std_deviation = 'cadphr-prq_std'
                df_prq[std_deviation] = df_prq['obs'].apply(
                    lambda x: calculate_std_with_sliding_window(x, sliding_window))
                df_prq['num_std_samples'] = df_prq[std_deviation].apply(lambda x: len(x))
                df_prq = df_prq[df_prq.num_std_samples >= 10]
                df_prq = df_prq.apply(
                    lambda x: extract_format_find_trend(x, trend_window), axis=1)
                processed_dataframes['prq_progression_df'] = [df_prq, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset2_V4_risk_clinical_risk_score_rule', 'ruleset2_V2_1_cad_risk_assessment_rule']:
            if cad_flag == 0:
                json_file_path_cad = os.path.join(json_folder, f"{'cadphr-cadrisk10'}.json")
                with open(json_file_path_cad, 'r') as json_file:
                    ehi_data = json.load(json_file)
                    ehi_dataframe_cad = pd.DataFrame(ehi_data)
                processed_cad_df, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    ehi_dataframe_cad,
                    "",
                    df_patient,
                    *common_params,
                    'cadphr-cadrisk10'
                )
                processed_dataframes["cad_df"] = [processed_cad_df, min_age, max_age, min_bmi, max_bmi]
                cad_flag = 1
        if rule_name in ['ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule']:
            vo2maxra_code = activity_code('resting', 'cadphr-vo2maxra')
            ecrfra_code = activity_code('resting', 'cadphr-ecrfra')
            vo2maxra_json_file_path = os.path.join(json_folder, "cadphr-vo2maxra.json")
            with open(vo2maxra_json_file_path, 'r') as vo2maxra_json_file:
                vo2maxra_data = json.load(vo2maxra_json_file)
                vo2maxra_dataframe = pd.DataFrame(vo2maxra_data)

            ecrfra_json_file_path = os.path.join(json_folder, "cadphr-ecrfra.json")
            with open(ecrfra_json_file_path, 'r') as ecrfra_json_file:
                ecrfra_data = json.load(ecrfra_json_file)
                ecrfra_dataframe = pd.DataFrame(ecrfra_data)

            processed_vo2maxra_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                vo2maxra_dataframe,
                vo2maxra_code,
                df_patient,
                *common_params,
                "cadphr-vo2maxra"
            )

            processed_ecrfra_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ecrfra_dataframe,
                ecrfra_code,
                df_patient,
                *common_params,
                "cadphr-ecrfra"
            )
            df_vo2max = processed_vo2maxra_dataframe[
                ['subject_reference', 'age', 'gender', 'cadphr-vo2maxra_grade', 'BMI']]
            df_ecrf = processed_ecrfra_dataframe[['subject_reference', 'age', 'gender', 'cadphr-ecrfra_grade', 'BMI']]
            merged_df = pd.merge(df_vo2max, df_ecrf, on=['subject_reference', 'age', 'gender'], how='outer')
            merged_df = merged_df.rename(columns={'cadphr-vo2maxra_grade': 'vo2max_grade'})
            merged_df['cadphr-vo2maxra_grade'] = merged_df['vo2max_grade'].fillna(merged_df['cadphr-ecrfra_grade'])
            merged_df['cadphr-vo2maxra_grade'] = merged_df['cadphr-vo2maxra_grade'].astype(int)
            merged_df.drop(labels=['BMI_x', 'vo2max_grade'], axis=1, inplace=True)
            merged_df.rename(columns={'BMI_y': 'BMI'}, inplace=True)
            processed_dataframes["vo2max_df"] = [merged_df, min_age, max_age, min_bmi, max_bmi]

        if rule_name in ['ruleset3_V4_emotional_influx_Stress_Disorder_rule']:
            ehi_perceived_stress = 'cadphr-pss4'
            ehi_stress_index = 'cadphr-sira'

            json_file_path_ps = os.path.join(json_folder, f"{ehi_perceived_stress}.json")
            json_file_path_si = os.path.join(json_folder, f"{ehi_stress_index}.json")

            with open(json_file_path_ps, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_ps = pd.DataFrame(ehi_data)

            with open(json_file_path_si, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_si = pd.DataFrame(ehi_data)

            processed_si_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_si,
                "",
                df_patient,
                *common_params,
                ehi_stress_index
            )

            processed_ps_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_ps,
                "",
                df_patient,
                *common_params,
                ehi_perceived_stress
            )

            merged_stress = pd.merge(processed_si_dataframe, processed_ps_dataframe[
                ['subject_reference', 'pss_value', 'cadphr-pss4_grade']], on='subject_reference',
                                     how='inner')
            processed_dataframes['stress_disorder_df'] = [merged_stress, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset1_V2_1_Heart_Rate_Recovery_rule']:
            json_file_name = 'cadphr-hrrra'
            json_file_path = os.path.join(json_folder, f"{json_file_name}.json")
            with open(json_file_path, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe = pd.DataFrame(ehi_data)

            processed_hrrra_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe,
                "",
                df_patient,
                *common_params,
                'cadphr-hrrra'
            )
            processed_dataframes['hrr_df'] = [processed_hrrra_dataframe, min_age, max_age, min_bmi, max_bmi]

        if rule_name in ['ruleset2_V2_1_osa_risk_assessment_rule']:
            json_file_path_osa = os.path.join(json_folder, f"{'cadphr-osariskscore'}.json")

            with open(json_file_path_osa, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_osa = pd.DataFrame(ehi_data)

            processed_osa_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_osa,
                "",
                df_patient,
                *common_params,
                'cadphr-osariskscore'
            )
            processed_dataframes['osa_df'] = [processed_osa_dataframe, min_age, max_age, min_bmi, max_bmi]

        if rule_name in ['ruleset2_V2_1_diabetes_risk_assessment_rule']:
            json_file_path_drs = os.path.join(json_folder, f"{'cadphr-diabetesriskscore'}.json")

            with open(json_file_path_drs, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_drs = pd.DataFrame(ehi_data)

            processed_drs_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_drs,
                "",
                df_patient,
                *common_params,
                'cadphr-diabetesriskscore'
            )
            processed_dataframes['drs_df'] = [processed_drs_dataframe, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule',
                         'ruleset3_V4_emotion_influx_depression_early_symptom_rule',
                         'ruleset3_V2_1_emotional_variability_rule']:
            json_file_path_emotion = os.path.join(json_folder, f"{'cadphr-emotionmeasure'}.json")
            na_code = activity_code('resting', 'cadphr-na')
            json_file_path_na = os.path.join(json_folder, f"{'cadphr-na'}.json")
            with open(json_file_path_emotion, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_ev = pd.DataFrame(ehi_data)
                ehi_dataframe_ei = pd.DataFrame(ehi_data)
                ehi_dataframe_einstability = pd.DataFrame(ehi_data)
            with open(json_file_path_na, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_na = pd.DataFrame(ehi_data)

            if rule_name == 'ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule':
                processed_einstability_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    ehi_dataframe_einstability,
                    "",
                    df_patient,
                    *common_params,
                    'cadphr-emotioninstability'
                )
                processed_dataframes['emotion_instability_df'] = [processed_einstability_dataframe, min_age, max_age,
                                                                  min_bmi, max_bmi]

            elif rule_name == 'ruleset3_V4_emotion_influx_depression_early_symptom_rule':
                processed_ei_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    ehi_dataframe_ei,
                    "",
                    df_patient,
                    *common_params,
                    'cadphr-emotioninertia'
                )
                processed_na_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    ehi_dataframe_na,
                    na_code,
                    df_patient,
                    *common_params,
                    'cadphr-na'
                )
                processed_ei_dataframe = pd.merge(processed_ei_dataframe,
                                                  processed_na_dataframe[['subject_reference', 'cadphr-na_grade']],
                                                  on='subject_reference', how='inner')
                processed_dataframes['emotion_inertia_df'] = [processed_ei_dataframe, min_age, max_age, min_bmi,
                                                              max_bmi]

            elif rule_name == 'ruleset3_V2_1_emotional_variability_rule':
                processed_ev_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                    ehi_dataframe_ev,
                    "",
                    df_patient,
                    *common_params,
                    'cadphr-emotionvariability'
                )
                processed_dataframes['emotion_variability_df'] = [processed_ev_dataframe, min_age, max_age, min_bmi,
                                                                  max_bmi]

        if rule_name in ['ruleset5_V4_progression_resprate_rule']:
            json_file_path = os.path.join(json_folder, f"{'cadphr-resprate'}.json")
            with open(json_file_path, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_rr = pd.DataFrame(ehi_data)

            processed_rr_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_rr,
                '9303-9',
                df_patient,
                *common_params,
                'cadphr-resprate'
            )

            processed_rr_dataframe = processed_rr_dataframe.apply(
                lambda x: extract_format_find_trend(x, trend_window), axis=1)
            processed_dataframes['resprate_progression_df'] = [processed_rr_dataframe, min_age, max_age, min_bmi,
                                                               max_bmi]

        if rule_name in ['ruleset2_V4_risk_clinical_risk_score_rule']:
            json_file_path = os.path.join(json_folder, f"{'cadphr-rcvage'}.json")
            with open(json_file_path, 'r') as json_file:
                ehi_data = json.load(json_file)
                ehi_dataframe_cad_Hage = pd.DataFrame(ehi_data)

            processed_cad_Hage_dataframe, min_date, max_date, min_age, max_age, min_bmi, max_bmi = process(
                ehi_dataframe_cad_Hage,
                'ehi-risk',
                df_patient,
                *common_params,
                'cadphr-rcvage'
            )
            processed_cad_dataframe = processed_dataframes['cad_df'][0]
            merged_df = pd.merge(processed_cad_Hage_dataframe[['subject_reference', 'cadphr-rcvage_value']],
                                 processed_cad_dataframe[
                                     ['subject_reference', 'cadphr-cadrisk10_value', 'gender', 'age']],
                                 on='subject_reference', how='inner')
            processed_dataframes['cad_heartage_df'] = [merged_df, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset1_V4_cardio_respiratory_fitness_rule']:
            data = pd.read_json('data_processing/configuration.json', typ='series')
            cvreactivity_features = data["cvreactivity_features"]
            sbp_json_file_path = os.path.join(json_folder, "cadphr-bloodpressure.json")
            prq_json_file_path = os.path.join(json_folder, "cadphr-prq.json")
            heartrate_json_file_path = os.path.join(json_folder, "cadphr-heartrate.json")

            with open(sbp_json_file_path, 'r') as sbp_json_file:
                sbp_data = json.load(sbp_json_file)
                df_sbp = pd.DataFrame(sbp_data)

            with open(prq_json_file_path, 'r') as prq_json_file:
                prq_data = json.load(prq_json_file)
                df_prq = pd.DataFrame(prq_data)

            with open(heartrate_json_file_path, 'r') as heartrate_json_file:
                heartrate_data = json.load(heartrate_json_file)
                df_hr = pd.DataFrame(heartrate_data)

            print("cvreactivity functions start from here:\n")
            df_sbp = extract_sbp_new(df_sbp)
            print("Common Parameters are", *(com_par for com_par in common_params))
            df_prq = extract_new_ehi(df_prq, common_params[0], common_params[1])
            df_hr = extract_new_ehi(df_hr, common_params[0], common_params[1])

            min_date = min(df_hr['Date'])
            max_date = max(df_hr['Date'])
            min_age = min(df_patient['age'])
            max_age = max(df_patient['age'])
            min_bmi, max_bmi = min(df_patient['BMI']), max(df_patient['BMI'])

            # activity_codes
            # PRQ
            resting_prq = 'PHR-1001'
            postactivity_prq = 'PHR-1016'
            # HR
            resting_hr = '40443-4'
            postactivity_hr = '40442-6'
            # BP
            resting_bp = '85354-9'
            postactivity_bp = '88346-2'

            df_react_hr = reactivity_process_df(df_hr, 'cadphr-heartrate', resting_hr, postactivity_hr)
            df_react_sbp = reactivity_process_df(df_sbp, 'cadphr-bloodpressure', resting_bp, postactivity_bp)
            df_react_prq = reactivity_process_df(df_prq, 'cadphr-prq', resting_prq, postactivity_prq)
            react_features = ehi_input(cvreactivity_features)
            lst = lst_of_df_cvreact(cvreactivity_features, df_react_hr, df_react_sbp, df_react_prq)
            df_final = final_cv_reactivity(lst, react_features)
            df_final = pd.merge(df_final, df_patient, on='subject_reference', how='inner')
            processed_dataframes['cvreactivity'] = [df_final, min_age, max_age, min_bmi, max_bmi]
        if rule_name in ['ruleset6_V4_combinatorial_rules_cardiac_power_index_rule']:
            data = pd.read_json('data_processing/configuration.json', typ='series')
            k_const = data["k_constant"]
            bp_json_file_path = os.path.join(json_folder, "cadphr-bloodpressure.json")
            with open(bp_json_file_path, 'r') as bp_json_file:
                bp_data = json.load(bp_json_file)
                bp_dataframe = pd.DataFrame(bp_data)
            hr_json_file_path = os.path.join(json_folder, "cadphr-heartrate.json")
            with open(hr_json_file_path, 'r') as json_file:
                hr_data = json.load(json_file)
                hr_dataframe = pd.DataFrame(hr_data)
            df_weight = pd.read_json('json_folder/cadphr-bodyweight.json')
            df_height = pd.read_json('json_folder/cadphr-bodyheight.json')
            df_patient = pd.read_json('patient_new.json')
            df_final, min_date, max_date, min_age, max_age, min_bsa, max_bsa = cardiac_index(bp_dataframe, bp_dataframe, hr_dataframe,
                                                                         df_patient, df_height, df_weight, k_const, gender, s_age, e_age, common_params[0], common_params[1])
            processed_dataframes['cpi_df'] = [df_final, min_age, max_age, min_bsa, max_bsa]
    return processed_dataframes, min_date, max_date
