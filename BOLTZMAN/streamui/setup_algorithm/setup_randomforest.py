from io import StringIO
import pandas as pd
import streamlit as st

from StorageManage import minio_client, DS_BUCKET_NAME

st.cache_resource
def select_features(file_name):
    object_contents = minio_client.get_object(DS_BUCKET_NAME, file_name)
    object_data = object_contents.data
    object_str = str(object_data, 'utf-8')
    data = StringIO(object_str)
    dataset_df = pd.read_csv(data)

    column_list = dataset_df.columns.tolist()
    sel_input = st.multiselect("Input field:", column_list)
    filter_in_columns = [col for col in column_list if col in sel_input]
    output_can_list = list(set(column_list) - set(filter_in_columns))
    filter_out_columns = st.selectbox("Output field:", output_can_list)

    review = st.button("Review")
    print("Review ", review)
    if review:
        if (len(filter_in_columns) > 0) and (len(filter_out_columns)) > 0:
            in_df = dataset_df[filter_in_columns]
            out_df = dataset_df[filter_out_columns]
            rfc_df = pd.concat([in_df, out_df], axis=1)
            st.dataframe(rfc_df)
            return {'in':filter_in_columns, 'out':filter_out_columns}

def select_randomforest_algorithm_options(file_name):
    numtree = st.number_input('Numbers of Tree', min_value=1, max_value=100, value=5, placeholder="Type a number..." )
    print('Number of tree', numtree)
    max_dept = st.number_input('Max Tree depth', value=3, placeholder="Type a number...")
    print('Max Tree Depth', max_dept)
    max_features = st.number_input('Max Features', value=3, placeholder="Type a number...")
    max_leaf_nodes = st.number_input('Max_Leaf_Nodes', value=3, placeholder="Type a number...")
    min_samples_leaf = st.number_input('Min Samples Leaf', value=3, placeholder="Type a number...")
    min_samples_split = st.number_input('Min Samples Split', value=3, placeholder="Type a number...")
    min_weight_fraction = st.number_input('Min Weight Fraction', value=3, placeholder="Type a number...")
    n_estimators = st.number_input('Numbers Estimators', value=10, placeholder="Type a number...")
    test_size = st.number_input('Test Size', value=10, placeholder="Type a number...")
    fit_parameter = {
        'numtree': numtree,
        'max_dept': max_dept,
        'max_features': max_features,
        'max_leaf_nodes': max_leaf_nodes,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'min_weight_fraction': min_weight_fraction,
        'n_estimators': n_estimators,
        'test_size': test_size
    }
    btn_fit = st.form_submit_button("Fit")
    if btn_fit:
        return fit_parameter



