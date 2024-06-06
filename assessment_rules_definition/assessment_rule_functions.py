def ruleset1_V2_1_Fitness_VO2max_Fatness_BMI_rule(row):
    KF_flag = 0

    if row['BMI'] >= 30:
        assessment = 'RF_(obese)_BMI>=30:tomato'
    elif 25 <= row['BMI'] < 30 and row['cadphr-vo2maxra_grade'] <= 2:
        assessment = 'RF_(overweight+unfit)_25<BMI<30 VO2max<2:orange'
        KF_flag = 1
    elif row['BMI'] < 25 and row['cadphr-vo2maxra_grade'] <= 2:
        assessment = 'RF_(normalweight+unfit)_BMI<25 VO2max<2:sandybrown'
        KF_flag = 1
    elif 25 <= row['BMI'] < 30 and row['cadphr-vo2maxra_grade'] >= 3:
        assessment = 'PF_(overweight+fit)_25<BMI<30 VO2max>3:lightgreen'
        KF_flag = 1
    elif row['BMI'] < 25 and row['cadphr-vo2maxra_grade'] >= 3:
        assessment = 'PF_(normalweight+fit)_BMI<25 VO2max>=3:green'
        KF_flag = 1
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset1_V2_1_Heart_Rate_Recovery_rule(row):
    if row['cadphr-hrrra_grade'] == 1:
        return 'RF_(unfit)_HRR=1:tomato'
    elif row['cadphr-hrrra_grade'] == 2:
        return 'PF_(fit)_HHR=2:lightgreen'


def ruleset2_V2_1_cad_risk_assessment_rule(row):
    if row['cadphr-cadrisk10_grade'] == 1:
        return 'RF_CADRisk=1:tomato'
    elif row['cadphr-cadrisk10_grade'] == 2:
        return 'RF_CADRisk=2:orange'
    elif row['cadphr-cadrisk10_grade'] == 4:
        return 'PF_CADRisk=4:lightgreen'
    elif row['cadphr-cadrisk10_grade'] == 5:
        return 'PF_CADRisk=5:green'
    else:
        return 'ICF:lightgrey'


def ruleset2_V2_1_diabetes_risk_assessment_rule(row):
    if row['cadphr-diabetesriskscore_grade'] == 1:
        return 'RF_DRS=1:orange'
    elif row['cadphr-diabetesriskscore_grade'] == 3:
        return 'PF_DRS=3:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset2_V2_1_osa_risk_assessment_rule(row):
    if row['cadphr-osariskscore_grade'] == 1:
        return 'RF_OSA=1:orange'
    elif row['cadphr-osariskscore_grade'] == 2:
        return 'PF_OSA=2:lightgreen'


def ruleset3_V2_1_emotional_variability_rule(row):
    KF_flag = 0
    if row['variability_latest_grade'] == 1:
        assessment = 'RF_EV_Grade=1:tomato'
    elif row['variability_latest_grade'] == 3:
        assessment = 'PF_EV_Grade=3:lightgreen'
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset4_V2_1_extreme_Heart_Rate_rule(row):
    if row['Latest_HR_Value'] >= 100 and row['cadphr-heartrate_grade'] == '1':
        return 'RF_HR>100; HR_Grade=1:tomato'
    elif row['Latest_HR_Value'] < 35 and row['cadphr-heartrate_grade'] == '1':
        return 'RF_HR<35; HR_Grade=1:sandybrown'
    elif row['Latest_HR_Value'] >= 90 and row['cadphr-heartrate_grade'] == '2':
        return 'RF_HR>90; HR_Grade=2:orange'
    elif row['Latest_HR_Value'] < 45 and row['cadphr-heartrate_grade'] == '1':
        return 'RF_HR<45; HR_Grade=1:peachpuff'
    elif row['cadphr-heartrate_grade'] == '5':
        return 'PF_Optimal:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset4_V2_1_extreme_Heart_Rate_Variability_rule(row):
    if row['cadphr-hrv_grade'] == 1:
        return 'RF_HRV_Grade=1:orange'
    elif row['cadphr-hrv_grade'] == 3:
        return 'PF_HRV_Grade=3:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset4_V2_1_extreme_PRQ_rule(row):
    if row['Latest_PRQ_Value'] >= 8 and row['cadphr-prq_grade'] <= 2:
        return 'RF_PRQ>8; PRQ_Grade<2:orange'
    elif row['Latest_PRQ_Value'] < 2.5 and row['cadphr-prq_grade'] <= 2:
        return 'RF_PRQ<2.5; PRQ_Grade<2:peachpuff'
    elif row['cadphr-prq_grade'] == 5:
        return 'PF_PRQ_Grade=5:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset4_V3_1_extreme_DBP_rule(row):
    if row['cadphr-dbp_grade'] == 1 and row['DBP'] >= 120:
        return 'RF_DBP_Grade=1; DBP>=120:tomato'
    elif row['cadphr-dbp_grade'] == 1 and row['DBP'] < 50:
        return 'RF_DBP_Grade=1; DBP<50:orange'
    elif row['cadphr-dbp_grade'] == 2 and row['DBP'] >= 95:
        return 'RF_DBP_Grade=2; DBP>=95:peachpuff'
    elif row['cadphr-dbp_grade'] == 2 and row['DBP'] < 60:
        return 'RF_DBP_Grade=2; DBP<60:sandybrown'
    elif row['cadphr-dbp_grade'] == 3:
        return 'RF_DBP_Grade=3:orange'
    elif row['cadphr-dbp_grade'] == 5:
        return 'PF_DBP_Grade=5:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset4_V3_1_extreme_SBP_rule(row):
    if row['cadphr-sbp_grade'] == 1 and row['SBP'] >= 180:
        return 'RF_SBP_grade=1 SBP>=180:tomato'
    elif row['cadphr-sbp_grade'] == 1 and row['SBP'] < 70:
        return 'RF_SBP_grade=1 SBP<70:peachpuff'
    elif row['cadphr-sbp_grade'] == 2 and row['SBP'] > 140:
        return 'RF_SBP_grade=2 SBP>140:orange'
    elif row['cadphr-sbp_grade'] == 2 and row['SBP'] < 80:
        return 'RF_SBP_grade=2 SBP<80:sandybrown'
    elif row['cadphr-sbp_grade'] == 3 and row['SBP'] >= 130:
        return 'RF_SBP_grade=3 SBP>=130:orange'
    elif row['cadphr-sbp_grade'] == 3 and row['SBP'] < 90:
        return 'RF_SBP_grade=3 SBP<90:sandybrown'
    elif row['cadphr-sbp_grade'] == 5:
        return 'PF_SBP_grade=5:lightgreen'
    else:
        return 'ICF:lightgrey'


def ruleset5_V3_1_progression_dbp_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-dbp_grade'] <= 2 and row['DBP'] >= 95:
        assessment = 'RF_DBP_grade <= 2 and DBP>=140; tau>=0.8:orange'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-dbp_grade'] == 3:
        assessment = 'RF_DBP_grade = 3 and DBP>=140; tau>=0.8:sandybrown'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-dbp_grade'] == 4 and row['DBP'] >= 80:
        assessment = 'ICF:lightgrey'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-dbp_grade'] in [5]:
        assessment = 'PF_DBP_grade is 5; tau<-0.8:limegreen'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-dbp_grade'] in [4]:
        assessment = 'ICF:lightgrey'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset5_V3_1_progression_hrv_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-hrv_grade'] == 3:
        assessment = 'PF_HRV_grade is 3; tau>=0.8:limegreen'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-hrv_grade'] == 2:
        assessment = 'ICF:lightgrey'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-hrv_grade'] == 1:
        assessment = 'PF_HRV_grade is 1; tau<-0.8:tomato'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-hrv_grade'] == 2:
        assessment = 'PF_HRV_grade is 2; tau<-0.8:orange'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset5_V3_1_progression_rhr_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and (
            row['cadphr-heartrate_grade'] == 3 or row['cadphr-heartrate_grade'] == 4):
        assessment = 'PF_RHR_grade is 3 or 4; tau>=0.8:peachpuff'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and (
            row['cadphr-heartrate_grade'] == 4 or row['cadphr-heartrate_grade'] == 5):
        assessment = 'PF_RHR_grade is 4 or 5; tau<-0.8:limegreen'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and (
            row['cadphr-heartrate_grade'] == 1 or row['cadphr-heartrate_grade'] == 2):
        assessment = 'RF_RHR_grade is 1 or 2; tau>=0.8:orange'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset5_V3_1_progression_sbp_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-sbp_grade'] <= 2 and row['SBP'] >= 140:
        assessment = 'RF_SBP_grade <= 2 and SBP>=140; tau>=0.8:orange'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-sbp_grade'] == 3 and row['SBP'] >= 130:
        assessment = 'RF_SBP_grade = 3 and SBP>=140; tau>=0.8:sandybrown'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-sbp_grade'] == 4 and row['SBP'] >= 120:
        assessment = 'ICF:lightgrey'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-sbp_grade'] == 5:
        assessment = 'PF_SBP_grade is 5; tau<-0.8:limegreen'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-sbp_grade'] == 4:
        assessment = 'ICF:lightgrey'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset6_V3_1_combinatorial_hr_hrv_rule(row):
    KF_flag = 0

    if row['cadphr-heartrate_grade'] == '1' and row['cadphr-hrv_grade'] == 1:
        assessment = 'RF_HR_Grade=1 HRV_Grade=1:orange'
        KF_flag = 1
    elif (row['cadphr-heartrate_grade'] == '4' or row['cadphr-heartrate_grade'] == '5') and row['cadphr-hrv_grade'] == 3:
        assessment = 'PF_HR_Grade Protective HRV_Grade=3:limegreen'
        KF_flag = 1
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset6_V3_1_combinatorial_rules_SBP_DBP_rule(row):
    KF_flag = 0

    if row['SBP'] >= 130 and row['DBP'] < 85:
        assessment = 'RF_(isolated_systolic)_SBP>=130 DBP<85:orange'
        KF_flag = 1
    elif row['SBP'] < 130 and row['DBP'] >= 85:
        assessment = 'RF_(isolated_diastolic)_SBP<130 DBP>=85:sandybrown'
        KF_flag = 1
    elif row['cadphr-sbp_grade'] == 5 and row['cadphr-dbp_grade'] == 5:
        assessment = 'PF_(normal)_SBP_grade=5 DBP=85:limegreen'
        KF_flag = 1
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset1_V4_cardio_respiratory_fitness_rule(row):
    if row['conclusion'] == 'Exaggerated':
        return 'RF_Exaggerated:tomato'
    elif row['conclusion'] == 'Normal':
        return 'PF_Normal:limegreen'


def ruleset2_V4_risk_clinical_risk_score_rule(row):
    assessment = ""
    if row['cadphr-rcvage_value'] - row['age'] > 2 and row['cadphr-cadrisk10_value'] > 10:
        assessment = 'RF_cadRisk10>10; HeartAge-age>2:tomato'
    elif row['cadphr-rcvage_value'] - row['age'] > 1 and row['cadphr-cadrisk10_value'] < 10:
        assessment = 'RF_cadRisk10<10; HeartAge-age>1:orange'
    elif row['cadphr-rcvage_value'] - row['age'] <= 2 and row['cadphr-cadrisk10_value'] > 10:
        assessment = 'RF_cadRisk10>10; HeartAge-age<2:sandybrown'
    elif row['cadphr-rcvage_value'] - row['age'] <= 1 and row['cadphr-cadrisk10_value'] < 10:
        assessment = 'PF_cadRisk10<10; HeartAge-age<1:limegreen'
    return assessment


def ruleset3_V4_emotional_flux_Stress_Disorder_rule(row):
    KF_flag = 0

    if row['stress_index_interpretation'] <= "2" and row['cadphr-pss4_grade'] == 1:
        assessment = 'RF_Stress_Index<=2; Percieved=1:tomato'
        KF_flag = 1
    elif row['stress_index_interpretation'] == "3" and row['cadphr-pss4_grade'] == 3:
        assessment = 'PF_Stress_Index=3; Percieved=3:limegreen'
    elif row['stress_index_interpretation'] == "3" and row['cadphr-pss4_grade'] == 1:
        assessment = 'ICF:lightgrey'
    elif row['stress_index_interpretation'] <= "2" and row['cadphr-pss4_grade'] == 3:
        assessment = 'RF_Stress_Index<=2; Percieved=3:peachpuff'
        KF_flag = 1
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset3_V4_emotion_influx_anxiety_disorder_predictor_rule(row):
    KF_flag = 0

    if row['instability_latest_grade'] == 1 and 15 <= row['GAD7'] <= 21:
        assessment = 'RF_EI_Grade=1 15<=GAD7<=21:tomato'
        KF_flag = 1
    elif row['instability_latest_grade'] == 3 and 0 <= row['GAD7'] <= 9:
        assessment = 'PF_EI_Grade=3 0<=GAD7<=9:limegreen'
    elif row['instability_latest_grade'] == 3 and 15 <= row['GAD7'] <= 21:
        assessment = 'ICF:lightgrey'
    elif row['instability_latest_grade'] == 1 and 0 <= row['GAD7'] <= 9:
        assessment = 'RF_EI_Grade=1 0<=GAD7<=9:orange'
        KF_flag = 1
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset3_V4_emotion_influx_depression_early_symptom_rule(row):
    KF_flag = 0
    if row['inertia_latest_grade'] == 1 and row['cadphr-na_grade'] <= 2:
        assessment = 'RF_EInertia_Grade=1; NA_grade<=2:tomato'
        KF_flag = 1
    elif row['inertia_latest_grade'] == 3:
        assessment = 'PF_EInertia_Grade=3:limegreen'
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset5_V4_progression_resprate_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-resprate_grade'] in [4, 5]:
        assessment = 'PF_RR_grade is 4 or 5; tau>=0.8:limegreen'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-resprate_grade'] in [4, 5]:
        assessment = 'PF_RR_grade is 4 or 5; tau<-0.8:lightgreen'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-resprate_grade'] in [1, 2, 3]:
        assessment = 'RF_RR_grade is 1 or 2 or 3; tau>=0.8:sandybrown'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-resprate_grade'] in [1, 2, 3]:
        assessment = 'RF_RR_grade is 1 or 2 or 3; tau<-0.8:peachpuff'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset5_V4_progression_prq_rule(row):
    if row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-prq_grade'] in [4, 5]:
        assessment = 'PF_PRQ_grade is 4 or 5; tau>=0.8:limegreen'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-prq_grade'] in [4, 5]:
        assessment = 'PF_PRQ_grade is 4 or 5; tau<-0.8:lightgreen'
    elif row['latest_tau'] >= 0.8 and row['latest_p_value'] < 0.05 and row['cadphr-prq_grade'] in [1, 2, 3]:
        assessment = 'RF_PRQ_grade is 1 or 2 or 3; tau>=0.8:sandybrown'
    elif row['latest_tau'] <= -0.8 and row['latest_p_value'] < 0.05 and row['cadphr-prq_grade'] in [1, 2, 3]:
        assessment = 'RF_PRQ_grade is 1 or 2 or 3; tau<-0.8:peachpuff'
    else:
        assessment = 'ICF:lightgrey'
    return assessment


def ruleset6_V4_combinatorial_rules_PulsePressure_SBP_rule(row):
    KF_flag = 0

    if row['Pulse Pressure'] < 0.25 * row['SBP']:
        assessment = 'RF_PP<25% of SBP:sandybrown'
        KF_flag = 1
    elif row['Pulse Pressure'] > 60:
        assessment = 'RF_PP>60:peachpuff'
        KF_flag = 1
    elif row['cadphr-pulsepressure_grade'] == 3:
        assessment = 'PF_PP_Grade = 3:limegreen'
    else:
        assessment = 'ICF:lightgrey'

    return assessment, KF_flag


def ruleset6_V4_combinatorial_rules_cardiac_power_index_rule(row):
    KF_flag = 0

    if row['CPI'] < 0.44:
        assessment = 'RF_CPI<0.44:orange'
        KF_flag = 1
    else:
        assessment = 'ICF'
    return assessment, KF_flag


def ruleset6_V4_combinatorial_rules_target_HR_rule(row):
    assessment = ""
    KF_flag = 0

    if row['target_hr_lower'] <= row['Highest HR of User'] <= row['target_hr_upper']:
        assessment = 'PF_Max HR within safe range:limegreen'
    elif row['Highest HR of User'] >= row['target_hr_upper']:
        assessment = 'RF_Max HR beyond safe range:orange'
        KF_flag = 1
    elif row['Highest HR of User'] <= row['target_hr_lower']:
        assessment = 'ICF:lightgrey'
    return assessment, KF_flag


# assigning assessment and KF flag to each dataframe
def assign_assessment_rule_KF_flag(df, rule_name):
    result = df.apply(rule_name, axis=1)

    # Extracting the assessment and KF_flag values into separate Series
    assessment_series = result.apply(lambda x: x[0])
    KF_flag_series = result.apply(lambda x: x[1])

    # Assigning the Series to new columns in the DataFrame
    df['assessment'] = assessment_series
    df['KF_flag'] = KF_flag_series
    return df
