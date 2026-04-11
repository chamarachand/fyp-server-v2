from schemas.dr_schema import TabularData

# def preprocess_tabular(tabular_data: TabularData):
#     return {
#         "age": tabular_data.age,
#         # "sex": tabular_data.sex,
#         "dm_time": tabular_data.dm_time,
#         "alcohol_consumption": int(tabular_data.alcohol_consumption),
#         "smoking": int(tabular_data.smoking)
#     }

def preprocess_tabular(tabular_data: TabularData):
    return {
        "age": tabular_data.age,
        "dm_time": tabular_data.dm_time,
        "alcohol_consumption": int(tabular_data.alcohol_consumption),
        "smoking": int(tabular_data.smoking),
        "age_x_dm_time": tabular_data.age * tabular_data.dm_time  # <-- Calculated dynamically
    }
