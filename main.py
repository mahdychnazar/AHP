import pandas as pd
import numpy as np
import streamlit as st

import io

random_consistency_values = [0.0, 0.0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.51, 1.48, 1.56, 1.57, 1.59]


def main():

    st.sidebar.title("Insert Data")

    with st.form("my-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("upload file")
        submitted = st.form_submit_button("submit")


    if uploaded_file is not None and submitted:
        target, criteria_list, options_list, weights, options_weights_list = data_input_from_file(uploaded_file)
        #print("*******************************************************************")
    else:
        target, criteria_list, options_list, weights, options_weights_list = data_input()
        #print(st.session_state)
    weights_list = options_weights_list.copy()
    weights_list.insert(0, weights)
    filename = "input.xlsx"
    dfs_to_excel(weights_list, filename)

    col1, col2 = st.columns(2)

    with col1:
        st.button("Calculate", key="calculate_btn", type="primary")

    with col2:
        st.download_button("Export input", open(filename, "rb"), file_name=target + "_input.xlsx")
    st.columns(1)
    st.divider()

    if st.session_state["calculate_btn"]:
        #print(options_weights_list)
        priorities_tables(weights, options_weights_list)

    return


def dfs_to_excel(dfs_list, filename):

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    dfs_list[0].to_excel(writer, sheet_name="criteria")
    for df, op in zip(dfs_list[1:], dfs_list[0].columns):
        df.to_excel(writer, sheet_name=op)
    writer.close()


def data_input_from_file(uploaded_file):

    # To read file as bytes:
    bytes_data = io.BytesIO(uploaded_file.getvalue())

    sheet_names = pd.ExcelFile(bytes_data).sheet_names

    target = uploaded_file.name.removesuffix("_input.xlsx")

    for sheet in sheet_names:
        st.session_state[sheet] = pd.read_excel(bytes_data, index_col=0, sheet_name=sheet)

    options_list = st.session_state[sheet_names[1]].columns.tolist()

    criteria_list = st.session_state[sheet_names[0]].columns.tolist()

    st.session_state.criteria_list = criteria_list

    st.session_state.options_list = options_list

    return data_input(target=target)

def data_input(target=None):


    if target is not None:
        target = st.sidebar.text_input("Set your goal", value=target)
    else:
        target = st.sidebar.text_input("Set your goal")


    if "criteria_list" not in st.session_state:
        criteria_number = st.sidebar.number_input("Set number of criteria", min_value=1, max_value=14)
    else:
        criteria_number = st.sidebar.number_input("Set number of criteria", min_value=1, max_value=14, value=len(st.session_state.criteria_list))

    if "options_list" not in st.session_state:
        option_number = st.sidebar.number_input("Set number of options", min_value=1, max_value=14)
    else:
        option_number = st.sidebar.number_input("Set number of options", min_value=1, max_value=14, value=len(st.session_state.options_list))

    st.session_state.criteria_list = set_criteria(criteria_number)

    st.session_state.options_list = set_options(option_number)

    criteria_weights = weights_input("criteria_weights", st.session_state.criteria_list)

    options_weights_list = []

    for criteria in st.session_state.criteria_list:
        options_weights_list.append(weights_input(criteria, st.session_state.options_list))

    return target, st.session_state.criteria_list, st.session_state.options_list, criteria_weights, options_weights_list


def set_criteria(criteria_number):
    criteria_list = []
    st.sidebar.title("Set criteria")
    if "criteria_list" not in st.session_state:
        for i in range(0, criteria_number):
            criteria_list.append(st.sidebar.text_input("creteria" + str(i), value=i))
    else:
        for i in range(0, criteria_number):
            if i < len(st.session_state.criteria_list):
                criteria_list.append(st.sidebar.text_input("creteria" + str(i), value=st.session_state.criteria_list[i]))
            else:
                criteria_list.append(st.sidebar.text_input("creteria" + str(i), value=i))


    return criteria_list


def set_options(option_number):
    option_list = []
    st.sidebar.title("Set options")
    if "options_list" not in st.session_state:
        for i in range(0, option_number):
            option_list.append(st.sidebar.text_input("option" + str(i), value=i))
    else:
        for i in range(0, option_number):
            if i < len(st.session_state.options_list):
                option_list.append(st.sidebar.text_input("option" + str(i), value=st.session_state.options_list[i]))
            else:
                option_list.append(st.sidebar.text_input("option" + str(i), value=i))

    return option_list


def weights_input(matrix_name, criterias):
    st.write(matrix_name)

    criteria_weights = np.ones((len(criterias), len(criterias)))
    df = pd.DataFrame(criteria_weights, index=criterias, columns=criterias).astype('float')

    if matrix_name not in st.session_state:
        st.session_state[matrix_name] = df

    if (df.shape != st.session_state[matrix_name].shape)\
            or df.columns.tolist() != st.session_state[matrix_name].columns.tolist():
        st.session_state[matrix_name] = pd.DataFrame(st.session_state[matrix_name], index=criterias, columns=criterias)
        st.session_state[matrix_name] = st.session_state[matrix_name].fillna(value=1)

    col_config = {}

    for name in criterias:
        col_config[name] = st.column_config.NumberColumn(disabled=None, required=True, min_value=(1 / 9), max_value=9)

    weights = st.data_editor(st.session_state[matrix_name],\
                             key=matrix_name + "_data_editor",\
                             column_config=col_config, \
                             use_container_width=True,\
                             on_change=adjust_weights,\
                             args=(matrix_name, criterias))

    st.markdown("Consistency value: " + str(compute_consistency(weights)))

    st.divider()
    return weights


def adjust_weights(matrix_name, criterias_names):
    state = st.session_state[matrix_name + "_data_editor"]

    for index, updates in state["edited_rows"].items():
        index = criterias_names[index]
        for key, value in updates.items():
            if key == index:
                st.session_state[matrix_name].loc[index, key] = 1
            else:
                st.session_state[matrix_name].loc[index, key] = value
                st.session_state[matrix_name].loc[key, index] = 1/value
    return


def compute_consistency(weights):
    length = weights.shape[0]
    if length <= 2:
        return 0.0
    weights_sum_vector = weights.sum(axis=0)
    weights_priority_vector_normalized = priority_vector_normalized(weights, length)
    lambda_max = weights_sum_vector.mul(weights_priority_vector_normalized).sum()
    consistency = (((lambda_max - length) / (length - 1)) / random_consistency_values[length-1])
    return consistency


def priority_vector_normalized(weights, length):
    weights_priority_vector = weights.product(axis=1).map(lambda x: x**(1/length))
    weights_priority_vector_normalized = weights_priority_vector.map(lambda x: x / weights_priority_vector.sum(axis=0))
    return weights_priority_vector_normalized


def priorities_tables(criteria_weights, options_weights_list):
    option_priorities = pd.DataFrame(index=options_weights_list[0].columns, columns=criteria_weights.columns)

    for criteria in criteria_weights.columns:
        option_priorities[criteria] = priority_vector_normalized(options_weights_list[criteria_weights.columns.get_loc(criteria)], \
                                                 len(options_weights_list[criteria_weights.columns.get_loc(criteria)]))

    criteria_priorities = priority_vector_normalized(criteria_weights, len(criteria_weights))
    st.markdown("Пріоритети альтернатив")
    st.dataframe(option_priorities)
    st.markdown("Пріоритети критеріїв")
    st.dataframe(pd.DataFrame(criteria_priorities).transpose())
    global_priorities = option_priorities.dot(criteria_priorities)
    st.markdown("Глобальні пріоритети")
    st.dataframe(global_priorities)
    return

main()
