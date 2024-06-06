import warnings

warnings.filterwarnings("ignore")


def activity_code(activity_value, ehi_value):
    if ehi_value == "cadphr-heartrate":
        if activity_value == "resting":
            code = "40443-4"
        else:
            code = "40442-6"
        return code
    elif ehi_value == "cadphr-hrv":
        if activity_value == "resting":
            code = "80404-7"
        else:
            code = "PHR-1015"
        return code
    elif ehi_value == "cadphr-pa":
        if activity_value == "resting":
            code = "PHR-1009"
        else:
            code = "PHR-1009"
        return code
    elif ehi_value == "cadphr-na":
        if activity_value == "resting":
            code = "PHR-1010"
        else:
            code = "PHR-1010"
        return code
    elif ehi_value == "cadphr-prq":
        if activity_value == "resting":
            code = "PHR-1001"
        else:
            code = "PHR-1001"
        return code
    elif ehi_value == "cadphr-sbp" or ehi_value == "cadphr-dbp":
        if activity_value == "resting":
            code = "85354-9"
        else:
            code = "88346-2"
        return code
    elif ehi_value == "cadphr-dprp":
        if activity_value == "resting":
            code = "PHR-1003"
        else:
            code = "PHR-1003"
        return code
