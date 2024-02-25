import io

from minio import Minio
import os
from io import StringIO
import pandas as pd
import streamlit as st

DS_BUCKET_NAME = 'dataset'

minio_host = os.environ.get('minio_host')
print('Storage Address:', minio_host)
minio_client = None

if minio_host == None:
    minio_client = Minio("localhost:9000", "dataignite", "dataignite", secure=False)
else:
    minio_client = Minio(minio_host, "dataignite", "dataignite", secure=False)

def get_files_minio(BUCKET_NAME):
    list_dataset = []
    for object in minio_client.list_objects(BUCKET_NAME):
        list_dataset.append(object.object_name)
    return list_dataset

@st.cache_data
def load_dataset_minio(BUCKET_NAME, data_name, limit_rows=None):
    object_contents = minio_client.get_object(BUCKET_NAME, data_name)
    object_data = object_contents.data
    _, file_type = os.path.splitext(data_name)
    if 'xls' in file_type:
        dataset_df = pd.read_excel(object_data)
    elif 'csv' or 'csv' in file_type:
        object_str = str(object_data, 'utf-8')
        data = StringIO(object_str)
        if limit_rows != None:
            dataset_df = pd.read_csv(data, nrows=limit_rows)
        else:
            dataset_df = pd.read_csv(data)
    elif 'png' or 'jpg' in file_type:
        st.image(data_name, caption=f"{data_name}")
    else:
        st.error(f"처리 할 수 없는 데이터 입니다.")
        st.stop()
    return dataset_df

def update_object(new_df, file_name):
    new_df = new_df.reset_index(drop=True).to_csv(index=False).encode('utf-8')

    minio_client.put_object(
        DS_BUCKET_NAME,
        file_name,
        data=io.BytesIO(new_df),
        length=len(new_df),
        content_type='application/csv'
    )

def drop_dataset(BUCKET_NAME, file_name):
    minio_client.remove_object(BUCKET_NAME, file_name)

def is_fileobj_inbucket(BUCKET_NAME, file_name):
    obj_list = minio_client.list_objects(BUCKET_NAME)
    for obj in obj_list:
        if file_name == obj.object_name:
            return True
    return False