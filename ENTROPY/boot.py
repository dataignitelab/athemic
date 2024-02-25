import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from streamlit import session_state as session
import pandas as pd
from urllib.parse import quote_plus
from StorageManage.minio import update_object, is_fileobj_inbucket, DS_BUCKET_NAME
from StorageManage.redis import redis_mgr
from pymongo import MongoClient
import seaborn as sns
import itertools
import socket
import logging
import boto3
import logstash
from trino.dbapi import connect as trino_conn
from io import StringIO
import io
from botocore.exceptions import EndpointConnectionError
from dilogger import DIlogger
import sys
import socket

elk_address = os.environ.get('elk_address')

if elk_address is not None:
    address_list = [elk_address]
else:
    address_list = None

index = 'dataignite-entropy-log'
elk_logger = DIlogger(address_list = address_list, index = index, level="DEBUG")
elk_logger.info({sys._getframe().f_code.co_name:f"Host name:{socket.gethostname()}"})

mongo_host = os.environ.get('mongo_host')
mongo_user = os.environ.get('mongo_user')
mongo_passwd = os.environ.get('mongo_passwd')
auth_source = os.environ.get('auth_source')
if mongo_host is None:
    mongo_host = 'localhost'
if mongo_user is None:
    mongo_user = 'dataignite'
if mongo_passwd is None:
    mongo_passwd = 'dataignite'
if auth_source is None:
    auth_source = 'datastore'
mongo_client = MongoClient(host = mongo_host,
                     username=mongo_user,
                     password=mongo_passwd,
                     authSource=auth_source,
                     authMechanism='SCRAM-SHA-256')

mongo_db = mongo_client.get_database(auth_source)
try:
    col_list = mongo_db.list_collections()
    for col_name in col_list:
        print(col_name)
except Exception as e:
    print(str(e))
    st.error('MongoDB Error')
    st.error(str(e))
    sys.exit(1)


def upsert_sessiondb_redis(DPM_SESSION, key, value):
    if DPM_SESSION is None:
        DPM_SESSION = {key: value}
    else:
        DPM_SESSION[key] = value
    json_dpm_session = json.dumps(DPM_SESSION, ensure_ascii=False).encode('utf-8')
    redis_mgr.set("DPM_SESSION", json_dpm_session)
    # redis_mgr.expire("DPM_SESSION", 60) #TTL 1min
    return DPM_SESSION


def get_IndexofWidget(DPM_SESSION, values, widget_type):
    if DPM_SESSION is None:
        return 0
    else:
        if widget_type in DPM_SESSION:
            session_widget_value = DPM_SESSION[widget_type]
            if session_widget_value is None:
                idx = 0
            else:
                if session_widget_value in values:
                    idx = values.index(session_widget_value)
                else:
                    idx = 0
        else:
            return 0
        return idx


def get_ValuesofWidget(DPM_SESSION, column_list, widget_type):
    if DPM_SESSION is None:
        return column_list
    else:
        if widget_type in DPM_SESSION:
            session_widget_value = DPM_SESSION[widget_type]
            if session_widget_value is None:
                return column_list
            else:
                return session_widget_value
        return column_list



def getall_collection_mongdo():
    coll_list = []
    for col in mongo_db.list_collections():
        coll_list.append(col)
    return coll_list


def get_collection_mongo(col_name):
    collection = mongo_db.get_collection(col_name)
    return collection

def get_alldocument_mongo(col_name):
    docs_list = []
    collection = mongo_db.get_collection(col_name)
    for doc in collection.find():
        docs_list.append(doc)
    return docs_list

def get_document_mongo(col_name, doc_key, doc_value):
    collection = mongo_db.get_collection(col_name)
    filter_doc= collection.find_one({doc_key: doc_value})
    return filter_doc


def get_inbound_systeminfos_mongo(sys_name):
    base_systeminfo_doc = get_document_mongo(f"etl_InBound_RDB", 'sys_name', sys_name)
    documents = get_alldocument_mongo(f"etl_InBound_RDB_{sys_name}")
    inb_store_info = []
    for document in documents:
        inb_store_info.append({
            'host_address': base_systeminfo_doc['host_address'],
            'host_port': base_systeminfo_doc['host_port'],
            'DB NAME': document['db_name'],
            'User name': document['username'],
            'password': document['password'],
            'db_type': document['db_type'],
            'Url': f"mysql://{document['username']}:{quote_plus(document['password'])}@{base_systeminfo_doc['host_address']}:{base_systeminfo_doc['host_port']}/{document['db_name']}"
            })
    return inb_store_info


def get_outbound_storeinfos_mongo(sys_name):
    base_systeminfo_doc = get_document_mongo("etl_OutBound_RDB", 'sys_name', sys_name)
    documents = get_alldocument_mongo(f"etl_OutBound_RDB_{sys_name}")
    outb_store_info = []
    for document in documents:
        outb_store_info.append({
            'host_address': base_systeminfo_doc['host_address'],
            'host_port': base_systeminfo_doc['host_port'],

            'DB NAME': document['db_name'],
            'User name': document['username'],
            'password': document['password'],
            'db_type': document['db_type'],
            'Url': f"mysql://{document['username']}:{quote_plus(document['password'])}@{base_systeminfo_doc['host_address']}:{base_systeminfo_doc['host_port']}/{document['db_name']}"
        })
    return outb_store_info


def get_mongo_collection(collection):
    doc_list = []
    collection = mongo_db.get_collection(collection)
    if collection is not None:
        for doc in collection.find():
            doc_list.append(doc)
    return doc_list


def upsert_mongo_doc(collection_name, filter, document):
    collection = mongo_db.get_collection(collection_name)
    if collection is None:
        mongo_db.create_collection(collection_name)
    collection = mongo_db.get_collection(collection_name)
    key = list(filter.keys())[0]
    value = filter[key]
    system_doc = collection.find_one(filter)
    if system_doc is None:
        document[key] = value
        collection.insert_one(document)
    else:
        update_filter = {key: value}
        collection.update_one(update_filter, {'$set': document})

def isexist_mongo_doc(collection_name, filter):
    collection_list = [ col for col in mongo_db.list_collection_names()]
    if collection_name in collection_list:
        docs = mongo_db.get_collection(collection_name)
        match_doc = docs.find_one(filter)
        if match_doc is not None:
            return True
        else:
            return False
    else:
        return False

def drop_doc_mongo(collection_name, drop_filter):
    collection = mongo_db.get_collection(collection_name)
    collection.delete_one(drop_filter)
    if isexist_mongo_doc(collection_name=collection_name, filter=drop_filter) == True:
        return False
    else:
        return True

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

def on_change_main_menu():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"],"MAIN_MENU",session.MAIN_MENU)

def on_change_migraion_path():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MIGRATION_CHANNEL", session.MIGRATION_CHANNEL)

def on_change_catalog():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "TRINO_CATALOGS", session.TRINO_CATALOGS)

def on_change_schema():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "TRINO_SCHEMA", session.TRINO_SCHEMA)

def on_change_table():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "TRINO_TABLES", session.TRINO_TABLES)

def on_change_inb_migraion_path():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "INB_MIGRATION_CHANNEL", session.INB_MIGRATION_CHANNEL)

def on_change_outb_migraion_path():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "OUTB_MIGRATION_CHANNEL",
                                                    session.OUTB_MIGRATION_CHANNEL)

def on_change_aws_svc_info():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_SVC",
                                                    session.AWS_SVC)
def on_change_aws_bucket_info():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_BUCKET",
                                                    session.AWS_BUCKET)

def on_change_aws_object_info():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_S3_OBJECT",
                                                    session.AWS_S3_OBJECT)

def on_change_migration_target_table():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MIG_DATA_NAME", session.MIG_DATA_NAME)

def on_change_mig_target_db_table_cols():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MIG_COLUMNS",
                                                    session.MIG_COLUMNS)

def on_change_em_interest_var():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EM_INTEREST_VAR",
                                                    session.EM_INTEREST_VAR)

def on_change_system_area():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "SYSTEM_AREA",
                                                    session.SYSTEM_AREA)

def on_change_storagetype():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "STORAGE_TYPE",
                                                    session.STORAGE_TYPE)

def on_change_cloudtype():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "CLOUD_TYPE",
                                                    session.CLOUD_TYPE)

def on_change_aws_svc_type():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_SVC_TYPE",
                                                    session.AWS_SVC_TYPE)

def on_change_bigdateengine_type():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "BIGDATAENGINE",
                                                    session.BIGDATAENGINE)

def on_change_db_vender():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DB_BANDER",
                                                    session.DB_BANDER)

def on_change_register_option():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "REGISTER_OPTION",
                                                    session.REGISTER_OPTION)

def on_change_bucket_option():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "BUCKET_NAME",
                                                    session.BUCKET_NAME)

def on_change_edit_systemname():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EDIT_SYSTEM_NAME",
                                                    session.EDIT_SYSTEM_NAME)

def on_change_edit_dbname():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EDIT_DB_NAME",
                                                    session.EDIT_DB_NAME)

def on_change_drop_systemname():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_SYSTEM_NAME",
                                                    session.DROP_SYSTEM_NAME)

def on_change_drop_dbname():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_SYSTEM_NAME",
                                                    session.DROP_SYSTEM_NAME)

def on_change_drop_bucketname():
    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_BUCKET_NAME",
                                                    session.DROP_BUCKET_NAME)


@st.cache_resource
def get_table(full_table, sel_input):
    mig_target_db_tables = full_table[sel_input]
    return mig_target_db_tables


#@st.cache_resource
def get_allcolumn_table(_db_conn, stored_default):
    sql = f"SELECT "
    sql = sql + ",".join(stored_default)
    sql = sql + f' FROM {session["DPM_SESSION"]["MIG_DATA_NAME"]}'
    all_col_table = _db_conn.query(sql)
    return all_col_table


def get_aws_s3_all_infos(bucket_name):
    s3_info = get_document_mongo(f"etl_AWS_S3", doc_key="bucket_name", doc_value=bucket_name)

    objects_name = get_aws_s3_object_list(s3_info['access_key'], s3_info['secret_key'],s3_info['aws_region'], bucket_name)
    s3_object_info = []
    for object_name in objects_name:
        s3_object_info.append({
            'region': s3_info['aws_region'],
            'Bucket Name': s3_info['bucket_name'],
            'Object Name': object_name,
            'Access Key': s3_info['access_key'],
            'Secretkey': s3_info['secret_key']})
    return s3_object_info

def get_aws_s3_bucket_list(access_key, secret_key, region):
    region_regex=re.compile('^us-gov-west-1|[a-z]{2}-[a-z]{4,9}-[1-6]$')
    if len(region) < 1:
        return '리전명 란이 비었습니다.'
    if region_regex.match(region) is None:
        return '리전명이 형식에 맞지 않습니다.'
    if len(access_key) < 1:
        return '액세스 키 란이 비었습니다.'
    if len(secret_key) < 1:
        return '시크릿 키 란이 비었습니다.'
    else:
        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
        try:
            bucket_info = s3.list_buckets()
            bucket_list = [buckets['Name'] for buckets in bucket_info['Buckets']]
            return bucket_list
        except EndpointConnectionError:
            return '해당 리전을 찾을 수 없습니다.'
        except Exception as e:
            st.error("입력된 정보가 올바르지 않습니다.")
            return e
def get_aws_s3_bucket_dict(access_key, secret_key, region):
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
    bucket_info = s3.list_buckets()
    bucket_dict = { "Bucket_Name": [buckets['Name'] for buckets in bucket_info['Buckets']]}
    return bucket_dict

def get_aws_s3_object_list(access_key, secret_key, region, bucket_name):
        s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
        obj_list = []
        contents = s3.list_objects(Bucket=bucket_name)['Contents']
        for content in contents:
            obj_list.append(content['Key'])
        return obj_list

def get_aws_s3_object_df(access_key, secret_key, region, bucket_name, object_name):
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
    obj = s3.get_object(Bucket=bucket_name, Key=object_name)
    s3_object_df = pd.read_csv(io.BytesIO(obj['Body'].read()), delimiter=',',encoding='utf-8')
    return s3_object_df

def update_aws_s3_bucket_object_info(region, access_key, secret_key, bucket_name, file_name):
    try:
        s3_object_df = get_aws_s3_object_df(access_key, secret_key, region, bucket_name, file_name)
        st.dataframe(s3_object_df)

        MIG_DATA_NAME = Path(file_name).stem
        target_columns = s3_object_df.columns.tolist()

        input_dname = ''
        input_ddesc = ''
        if MIG_DATA_NAME is not None:
            ds_store_collec = mongo_db.get_collection(f"ds_store_{MIG_DATA_NAME}")
            if ds_store_collec is not None:
                desc_docs = ds_store_collec.find()
                for desc_doc in desc_docs:
                    input_dname = desc_doc['ds_name']
                    input_ddesc = desc_doc['describe']
            else:
                input_dname = MIG_DATA_NAME
                input_ddesc = ''
        object_name = st.text_input(label="객체 이름", value=f"{MIG_DATA_NAME}.csv")
        data_name = st.text_input("데이터 이름", value=input_dname)
        data_describe = st.text_area("데이터 설명", value=input_ddesc)
        if data_describe is not None:
            if len(data_describe) > 0:
                ds_store_collec = mongo_db.get_collection(
                    f"ds_store_{MIG_DATA_NAME}")
                if ds_store_collec is None:
                    ds_store_collec = mongo_db.create_collection(
                        f"ds_store_{MIG_DATA_NAME}")
                    ds_store_collec.insert_one({"object_name": f"{object_name}",
                                                "ds_name": f"{data_name}",
                                                "describe": f"{data_describe}"})
                else:
                    ds_docs = ds_store_collec.find()
                    if ds_docs is not None:
                        for ds_doc in ds_docs:
                            ds_store_collec.delete_many({"ds_name": ds_doc["ds_name"]})
                        new_doc = {"object_name": f"{object_name}", "ds_name": f"{data_name}",
                                   "describe": f"{data_describe}"}
                        ds_store_collec.insert_one(new_doc)
                    else:
                        new_doc = {"object_name": f"{object_name}", "ds_name": f"{data_name}",
                                   "describe": f"{data_describe}"}
                        ds_store_collec.insert_one(new_doc)

        default_value = get_ValuesofWidget(session['DPM_SESSION'], target_columns, 'MIG_COLUMNS')
        if set(default_value).issubset(set(target_columns)) == True:
            stored_default = default_value
        else:
            stored_default = target_columns

        st.multiselect("대상 컬럼 선택",
                       target_columns,
                       default=stored_default,
                       key="MIG_COLUMNS",
                       on_change=on_change_mig_target_db_table_cols)
        sel_input = upsert_sessiondb_redis(session['DPM_SESSION'], 'MIG_COLUMNS',
                                           session.MIG_COLUMNS)
        sel_input = sel_input['MIG_COLUMNS']
        # object_name = st.text_input(label="객체 이름", value=f"{MIG_DATA_NAME}.csv")

        st.write(f"###### 데이터 사전 분석")
        if len(session["DPM_SESSION"]["MIG_COLUMNS"]) > 0:
            target_table = s3_object_df[sel_input]

            col_dtype, value_kinds = st.columns(2)
            with col_dtype:
                target_table_dtype = pd.DataFrame(target_table.dtypes, columns=['데이터 타입'])
                st.dataframe(target_table_dtype, width=400)
            with value_kinds:
                target_table_unique = pd.DataFrame(target_table.nunique(), columns=['컬럼별 값의 종류 수'])
                st.dataframe(target_table_unique, width=400)
            numeric_cols = target_table.select_dtypes([np.int64, np.float64]).columns
            integer_cols = list(target_table.select_dtypes([np.int64]).columns)

            st.write(f"###### 상관관계 차트")
            euclidean_corr_table = pd.DataFrame(target_table[numeric_cols].corr())
            st.dataframe(euclidean_corr_table, width=700)

            st.write(f"###### 상관관계 히트맵")
            fig, ax = plt.subplots(figsize=(12, 4))
            plt.title(f"Numeric column data correlation")
            sns.heatmap(euclidean_corr_table, ax=ax, annot=True, cmap='coolwarm', fmt=".2f")
            st.write(fig)

            st.info("Gausian Mixure Clustering을 이용한 학습 가능성 추정")
            st.info("Machine Learning 기법을 이용한 학습은 이곳에서 수행 할 수 없습니다.")
            agree_gmm = st.checkbox(label="가우시안 군집을 활용한 학습 가능성 변수 판단")
            if agree_gmm:
                opt_index = get_IndexofWidget(session["DPM_SESSION"], integer_cols, "EM_INTEREST_VAR")
                st.selectbox(
                    label=f"관심 변수",
                    options=integer_cols,
                    index=opt_index,
                    key="EM_INTEREST_VAR",
                    on_change=on_change_em_interest_var)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EM_INTEREST_VAR",
                                                                session.EM_INTEREST_VAR)

                em_interest_var = session["DPM_SESSION"]["EM_INTEREST_VAR"]
                data_cols = list(set(numeric_cols) - set(em_interest_var))

                interest_value = pd.DataFrame(target_table_unique.loc[em_interest_var])
                inter_values_cont = interest_value[em_interest_var].values.tolist()[0]

                cluster_col_table = target_table[data_cols]
                cluster_col_table['관심 변수의 값'] = target_table[em_interest_var]
                gmm = GaussianMixture(n_components=3, random_state=0)
                gmm_label = gmm.fit(cluster_col_table).predict(cluster_col_table)
                cluster_col_table['군집 번호'] = gmm_label
                gmm_target_cluster_df = pd.DataFrame(
                    {'count': cluster_col_table.groupby('관심 변수의 값')['군집 번호'].value_counts()}).reset_index()
                gmm_target_cluster_df.rename(columns={"count": "개체수"},inplace=True)
                st.dataframe(gmm_target_cluster_df, width=700, hide_index=True)

                filter_df = gmm_target_cluster_df.loc[gmm_target_cluster_df['관심 변수의 값'] == 0]
                if filter_df.shape[0] > inter_values_cont:
                    st.info("군집수가 과다 합니다. 다른 변수를 선택해보세요")
                else:
                    st.info("군집수와 관심변수 값의 수가 동일하여 중요 변수로 사용 가능합니다.")

            if st.button("복사"):
                update_object(target_table, object_name)
                if is_fileobj_inbucket(DS_BUCKET_NAME, object_name) == True:
                    st.write(f"ML Storage에 {object_name} 생성 완료")

    except Exception as e:
        print(e)
        st.error(str(e))


def analysis_migraion_inoutbound_data(bound, sys_name):
    # InOutBound에서 선택된 시스템의 등록된 모든 DB정보를 가져온다.
    # InOutBound Collectiond에 등록된 정보도 포함해야 함.
    if bound == 'In':
        db_infos = get_inbound_systeminfos_mongo(sys_name)
    elif bound == 'Out':
        db_infos = get_outbound_storeinfos_mongo(sys_name)

    st.write("### Machine Learning DataStorage에 생성할 DB 목록")

    if len(db_infos) == 0:
        st.warning("등록된 DB가 존재하지 않습니다.")
        st.warning("DB 등록을 통해 연계를 설정하세요.")
    else:
        db_infos_df = pd.DataFrame(data=db_infos)
        st.dataframe(db_infos_df, height=300)

        dbids = db_infos_df['DB NAME'].tolist()
        MIG_TARGET_DB_NAME = st.selectbox(label="DataBase를 선택하세요",
                                     options=dbids,
                                     index=0)
        try:
            sel_row = db_infos_df.loc[db_infos_df['DB NAME'] == MIG_TARGET_DB_NAME]
            url = sel_row['Url'].iloc[0]
            db_conn = st.connection(MIG_TARGET_DB_NAME, type="sql", url=url)
            tables_df = db_conn.query("SHOW TABLES")

            st.info(f"{url}에 연결 완료")
            tables_df.rename(columns={"Tables_in_kaggle": "테이블 목록"}, inplace=True)
            st.dataframe(tables_df, height=150, width=700, hide_index=True)

            taget_tables = tables_df['테이블 목록'].values.tolist()
            opt_index = get_IndexofWidget(session["DPM_SESSION"], taget_tables, "MIG_DATA_NAME")
            st.selectbox(label="대상 테이블 선택",
                         options=taget_tables,
                         index=opt_index,
                         key="MIG_DATA_NAME",
                         on_change=on_change_migration_target_table)
            session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MIG_DATA_NAME",
                                                            session.MIG_DATA_NAME)
            MIG_DATA_NAME = session["DPM_SESSION"]["MIG_DATA_NAME"]

            sql = f"SELECT * FROM {MIG_DATA_NAME} LIMIT 30"
            tables_df = db_conn.query(sql)
            target_columns = tables_df.columns.tolist()
            st.dataframe(tables_df.set_index(tables_df.columns[0]), width=700)

            input_dname = ''
            input_ddesc = ''
            if MIG_DATA_NAME is not None:
                ds_store_collec = mongo_db.get_collection(f"ds_store_{MIG_DATA_NAME}")
                if ds_store_collec is not None:
                    desc_docs = ds_store_collec.find()
                    for desc_doc in desc_docs:
                        input_dname = desc_doc['ds_name']
                        input_ddesc = desc_doc['describe']
                else:
                    input_dname = MIG_DATA_NAME
                    input_ddesc = ''

            object_name = st.text_input(label="객체 이름", value='{}.csv'.format(session["DPM_SESSION"]["MIG_DATA_NAME"]))
            data_name = st.text_input("데이터 이름", value=input_dname)
            data_describe = st.text_area("데이터 설명", value=input_ddesc)
            if data_describe is not None:
                if len(data_describe) > 0:
                    ds_store_collec = mongo_db.get_collection(
                        f"ds_store_{MIG_DATA_NAME}")
                    if ds_store_collec is None:
                        ds_store_collec = mongo_db.create_collection(
                            f"ds_store_{MIG_DATA_NAME}")

                        ds_store_collec.insert_one({"object_name": f"{object_name}",
                                                    "ds_name": f"{data_name}",
                                                    "describe": f"{data_describe}"})
                    else:
                        ds_docs = ds_store_collec.find()
                        if ds_docs is not None:
                            for ds_doc in ds_docs:
                                ds_store_collec.delete_many({"ds_name": ds_doc["ds_name"]})
                            new_doc = {"object_name": f"{object_name}","ds_name": f"{data_name}", "describe": f"{data_describe}"}
                            ds_store_collec.insert_one(new_doc)
                        else:
                            new_doc = {"object_name": f"{object_name}","ds_name": f"{data_name}", "describe": f"{data_describe}"}
                            ds_store_collec.insert_one(new_doc)

            default_value = get_ValuesofWidget(session['DPM_SESSION'], target_columns, 'MIG_COLUMNS')
            if set(default_value).issubset(set(target_columns)) == True:
                stored_default = default_value
            else:
                stored_default = target_columns
            full_table_df = get_allcolumn_table(db_conn, stored_default)
            st.multiselect("대상 컬럼 선택",
                           target_columns,
                           default=stored_default,
                           key="MIG_COLUMNS",
                           on_change=on_change_mig_target_db_table_cols)
            sel_input = upsert_sessiondb_redis(session['DPM_SESSION'], 'MIG_COLUMNS',
                                               session.MIG_COLUMNS)
            sel_input = sel_input['MIG_COLUMNS']


            st.write(f"###### 데이터 사전 분석")
            if len(session["DPM_SESSION"]["MIG_COLUMNS"]) > 0:
                target_table = get_table(full_table_df, sel_input)
                col_dtype, value_kinds = st.columns(2)
                with col_dtype:
                    target_table_dtype = pd.DataFrame(target_table.dtypes, columns=['데이터 타입'])
                    st.dataframe(target_table_dtype, width=400)
                with value_kinds:
                    target_table_unique = pd.DataFrame(target_table.nunique(), columns=['컬럼별 값의 종류 수'])
                    st.dataframe(target_table_unique, width=400)
                numeric_cols = target_table.select_dtypes([np.int64, np.float64]).columns
                integer_cols = list(target_table.select_dtypes([np.int64]).columns)

                st.write(f"###### 상관관계 차트")
                euclidean_corr_table = pd.DataFrame(target_table[numeric_cols].corr())
                st.dataframe(euclidean_corr_table)

                st.write(f"###### 상관관계 히트맵")
                fig, ax = plt.subplots(figsize=(12, 4))
                plt.title(f"Numeric column data correlation")
                sns.heatmap(euclidean_corr_table, ax=ax, annot=True, cmap='coolwarm', fmt=".2f")
                st.write(fig)

                st.info("Gausian Mixure Clustering을 이용한 학습 가능성 추정")
                st.info("Machine Learning 기법을 이용한 학습은 이곳에서 수행 할 수 없습니다.")
                agree_gmm = st.checkbox(label="가우시안 군집을 활용한 학습 가능성 변수 판단")
                if agree_gmm:
                    opt_index = get_IndexofWidget(session["DPM_SESSION"], integer_cols, "EM_INTEREST_VAR")
                    st.selectbox(
                        label=f"관심 변수",
                        options=integer_cols,
                        index=opt_index,
                        key="EM_INTEREST_VAR",
                        on_change=on_change_em_interest_var)
                    session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EM_INTEREST_VAR",
                                                                    session.EM_INTEREST_VAR)

                    em_interest_var = session["DPM_SESSION"]["EM_INTEREST_VAR"]
                    data_cols = list(set(numeric_cols) - set(em_interest_var))

                    interest_value = pd.DataFrame(target_table_unique.loc[em_interest_var])
                    inter_values_cont = interest_value[em_interest_var].values.tolist()[0]

                    cluster_col_table = target_table[data_cols]
                    cluster_col_table['관심 변수의 값'] = target_table[em_interest_var]
                    gmm = GaussianMixture(n_components=3, random_state=0)
                    gmm_label = gmm.fit(cluster_col_table).predict(cluster_col_table)
                    cluster_col_table['군집 번호'] = gmm_label
                    gmm_target_cluster_df = pd.DataFrame(
                        {'count': cluster_col_table.groupby('관심 변수의 값')['군집 번호'].value_counts()}).reset_index()
                    st.dataframe(gmm_target_cluster_df, hide_index=True, width=700)

                    filter_df = gmm_target_cluster_df.loc[gmm_target_cluster_df['관심 변수의 값'] == 0]
                    if filter_df.shape[0] > inter_values_cont:
                        st.info("군집수가 과다 합니다. 다른 변수를 선택해보세요")
                    else:
                        st.info("군집수와 관심변수 값의 수가 동일하여 중요 변수로 사용 가능합니다.")

                if st.button("복사"):
                    update_object(target_table, object_name)
                    if is_fileobj_inbucket(DS_BUCKET_NAME, object_name) == True:
                        st.write(f"ML Storage에 {object_name} 생성 완료")
        except Exception as e:
            print(e)
            st.error(str(e))

def keys_sysnameinb_from_mongo():
    keys = []
    docs = get_mongo_collection("etl_InBound_RDB")
    for doc in docs:
        keys.append(doc["sys_name"])
    return keys

def keys_sysnameoutb_from_mongo():
    keys = []
    docs = get_mongo_collection("etl_OutBound_RDB")
    for doc in docs:
        keys.append(doc["sys_name"])
    return keys

def get_etl_inbound_rdb_list():
    channel_col_name = f"etl_InBound_RDB"
    etlinbound_docs = get_alldocument_mongo(channel_col_name)
    inb_rpt_list = []
    for etlinb_doc in etlinbound_docs:
        sys_col_name = f"{channel_col_name}_{etlinb_doc['sys_name']}"
        etlinb_sys_docs = get_alldocument_mongo(sys_col_name)
        for etlinb_sys_doc in etlinb_sys_docs:
            inb_rpt_list.append({"채널": etlinb_doc["bound_area"],
                                 "시스템 이름": etlinb_doc["sys_name"],
                                 "Host Address": etlinb_doc["host_address"],
                                 "Host Port": etlinb_doc["host_port"],
                                 "데이터베이스 이름": etlinb_sys_doc["db_name"],
                                 "User Name": etlinb_sys_doc["username"],
                                 "Password": etlinb_sys_doc["password"],
                                 "Database 종류": etlinb_sys_doc["db_type"]
             })
    return inb_rpt_list

def get_etl_aws_s3_details_list():
    channel_col_name = f"etl_AWS_S3"
    etl_s3_docs = get_alldocument_mongo(channel_col_name)
    s3_rpt_list = []
    for etl_s3_doc in etl_s3_docs:
        bucket_name_col_name = f"{channel_col_name}_{etl_s3_doc['bucket_name']}"
        object_docs = get_alldocument_mongo(bucket_name_col_name)
        for object_doc in object_docs:
            s3_rpt_list.append({'리전':etl_s3_doc['aws_region'],
                                '버킷 이름': etl_s3_doc['bucket_name'],
                                '객체 이름': object_doc['object_name'],
                                'access key': etl_s3_doc['access_key'],
                                'secret key': etl_s3_doc['secret_key']})
    return s3_rpt_list

def get_etl_aws_s3_bucket_list():
    channel_col_name = f"etl_AWS_S3"
    etl_s3_docs = get_alldocument_mongo(channel_col_name)
    s3_bucket_list = []
    for etl_s3_doc in etl_s3_docs:
        s3_bucket_list.append({'Aws Region':etl_s3_doc['aws_region'],
                               '버킷 이름': etl_s3_doc['bucket_name'],
                               'Access Key': etl_s3_doc['access_key'],
                               'Secret Key': etl_s3_doc['secret_key'],
                               })
    return s3_bucket_list

def get_etl_aws_s3_bucket_object_list(bucket_name):
    channel_col_name = f"etl_AWS_S3_{bucket_name}"
    etl_s3_objects = get_alldocument_mongo(channel_col_name)
    s3_object_list = []
    for etl_s3_object in etl_s3_objects:
        s3_object_list.append(etl_s3_object['object_name'])
    return s3_object_list



def get_etl_outbound_rdb_list():
    channel_col_name = f"etl_OutBound_RDB"
    etloutb_docs = get_alldocument_mongo(channel_col_name)
    outb_rpt_list = []
    for etloutb_doc in etloutb_docs:
        sys_col_name = f"{channel_col_name}_{etloutb_doc['sys_name']}"
        etlinb_sys_docs = get_alldocument_mongo(sys_col_name)
        for etloutb_sys_doc in etlinb_sys_docs:
            outb_rpt_list.append({"채널": etloutb_doc["bound_area"],
                                 "시스템 이름": etloutb_doc["sys_name"],
                                 "Host Address": etloutb_doc["host_address"],
                                 "Host Port": etloutb_doc["host_port"],
                                 "데이터베이스 이름": etloutb_sys_doc["db_name"],
                                 "User Name": etloutb_sys_doc["username"],
                                 "Password": etloutb_sys_doc["password"],
                                 "Database 종류": etloutb_sys_doc["db_type"]
             })
    return outb_rpt_list

def get_etl_trino_catalogs_mongo():
    channel_col_name = f"etl_Trino"
    etltrino_docs = get_alldocument_mongo(channel_col_name)
    outb_rpt_list = []
    for etltrino_doc in etltrino_docs:
        outb_rpt_list.append({"Host Address": etltrino_doc["trino_ip"],
                              "Host Port": etltrino_doc["trino_port"],
                              "User Name": etltrino_doc["trino_user"],
                              "catalog": etltrino_doc["catalog"]
                              })
    return outb_rpt_list

@st.cache_resource
def get_schemas_in_catalog_df(trino_ip, trino_port, trino_user, catalog_nm):
    t_conn = trino_conn(host=trino_ip, port=trino_port, user=trino_user)
    cur = t_conn.cursor()
    cur.execute(f"SHOW SCHEMAS FROM {catalog_nm}")
    rows = cur.fetchall()
    queryResults_df = pd.DataFrame.from_records(rows, columns=[i[0] for i in cur.description])
    return queryResults_df

@st.cache_resource
def get_schemas_in_catalog_dict(trino_ip, trino_port, trino_user, catalog_nm):
    t_conn = trino_conn(host=trino_ip, port=trino_port, user=trino_user)
    cur = t_conn.cursor()
    cur.execute(f"SHOW SCHEMAS FROM {catalog_nm}")
    rows = cur.fetchall()
    queryResults_df = pd.DataFrame.from_records(rows, columns=[i[0] for i in cur.description])
    return queryResults_df.to_dict(orient='list')

@st.cache_resource
def get_table_in_trino_schema_dict(trino_ip, trino_port, trino_user, catalog_nm, schema):
    t_conn = trino_conn(host=trino_ip, port=trino_port, user=trino_user)
    cur = t_conn.cursor()
    cur.execute(f"SHOW TABLES FROM {catalog_nm}.{schema}")
    rows = cur.fetchall()
    queryResults_df = pd.DataFrame.from_records(rows, columns=[i[0] for i in cur.description])
    return queryResults_df.to_dict(orient='list')

@st.cache_resource
def run_trino_sql(trino_ip, trino_port, trino_user, trino_sql):
    t_conn = trino_conn(host=trino_ip, port=trino_port, user=trino_user)
    cur = t_conn.cursor()
    cur.execute(f"{trino_sql}")
    rows = cur.fetchall()
    queryResults_df = pd.DataFrame.from_records(rows, columns=[i[0] for i in cur.description])
    return queryResults_df


def regist_inout_rdb():
    if session["DPM_SESSION"]["STORAGE_TYPE"] == "관계형DB":
        st.write("등록된 저장소 목록")
        if session["DPM_SESSION"]["SYSTEM_AREA"] == "InBound":
            etl_inbound_list = get_etl_inbound_rdb_list()
            if len(etl_inbound_list) > 0:
                etl_inoutbound_df = pd.DataFrame(etl_inbound_list)
                st.dataframe(etl_inoutbound_df, height=300)
            else:
                etl_inoutbound_df = None
                st.info("등록된 저장소가 없습니다.")

        elif session["DPM_SESSION"]["SYSTEM_AREA"] == "OutBound":
            etl_outbound_list = get_etl_outbound_rdb_list()
            if len(etl_outbound_list) > 0:
                etl_inoutbound_df = pd.DataFrame(etl_outbound_list)
                st.dataframe(etl_inoutbound_df, height=300)
            else:
                etl_inoutbound_df = None
                st.info("등록된 저장소가 없습니다.")

        if etl_inoutbound_df is None:
            register_option = ["등록"]
        else:
            register_option = ["등록", "수정", "삭제"]
        opt_index = get_IndexofWidget(session["DPM_SESSION"], register_option, "REGISTER_OPTION")
        st.selectbox(label="등록/수정/삭제",
                     options=register_option,
                     index=opt_index,
                     key="REGISTER_OPTION",
                     on_change=on_change_register_option)
        session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "REGISTER_OPTION",
                                                        session.REGISTER_OPTION)

        if session["DPM_SESSION"]["REGISTER_OPTION"] == "등록":
            system_name = st.text_input("시스템 이름", "DP-Kaggle")
        elif session["DPM_SESSION"]["REGISTER_OPTION"] == "수정":
            if etl_inoutbound_df is not None:
                system_list = etl_inoutbound_df["시스템 이름"].tolist()
                opt_index = get_IndexofWidget(session["DPM_SESSION"], register_option, "EDIT_SYSTEM_NAME")
                st.selectbox(label="수정할 시스템을 선택하세요",
                             options=system_list,
                             index=opt_index,
                             key="EDIT_SYSTEM_NAME",
                             on_change=on_change_edit_systemname)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EDIT_SYSTEM_NAME",
                                                                session.EDIT_SYSTEM_NAME)

                dbname_list = etl_inoutbound_df["데이터베이스 이름"].tolist()
                opt_index = get_IndexofWidget(session["DPM_SESSION"], dbname_list, "EDIT_DB_NAME")
                st.selectbox(label="수정할 데이터베이스를 선택하세요",
                             options=dbname_list,
                             index=opt_index,
                             key="EDIT_DB_NAME",
                             on_change=on_change_edit_dbname)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EDIT_DB_NAME",
                                                                session.EDIT_DB_NAME)
        elif session["DPM_SESSION"]["REGISTER_OPTION"] == "삭제":
            if etl_inoutbound_df is not None:
                system_list = etl_inoutbound_df["시스템 이름"].tolist()
                opt_index = get_IndexofWidget(session["DPM_SESSION"], system_list, "DROP_SYSTEM_NAME")
                st.selectbox(label="시스템을 선택하세요",
                             options=system_list,
                             index=opt_index,
                             key="DROP_SYSTEM_NAME",
                             on_change=on_change_drop_systemname)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_SYSTEM_NAME",
                                                                session.DROP_SYSTEM_NAME)

                drop_system_name = session["DPM_SESSION"]["DROP_SYSTEM_NAME"]
                sel_row = etl_inoutbound_df.loc[(etl_inoutbound_df["시스템 이름"] == drop_system_name)]
                db_name_list = sel_row["데이터베이스 이름"].tolist()
                opt_index = get_IndexofWidget(session["DPM_SESSION"], db_name_list, "DROP_DB_NAME")
                st.selectbox(label="삭제할 데이터베이스 이름을 선택하세요",
                             options=db_name_list,
                             index=opt_index,
                             key="DROP_DB_NAME",
                             on_change=on_change_drop_dbname)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_DB_NAME",
                                                                session.DROP_DB_NAME)
                print(session["DPM_SESSION"]["DROP_SYSTEM_NAME"])
                print(session["DPM_SESSION"]["DROP_DB_NAME"])

        st.markdown("---")

        if session["DPM_SESSION"]["REGISTER_OPTION"] == "등록" or session["DPM_SESSION"]["REGISTER_OPTION"] == "수정":
            option_db_vender_list = ["MariaDB", "MySQL", "PostgreSQL"]
            opt_index = get_IndexofWidget(session["DPM_SESSION"], option_db_vender_list, "DB_VENDER")
            st.selectbox(label='{} 데이터 저장소 타입'.format(session["DPM_SESSION"]["SYSTEM_AREA"]),
                         options=option_db_vender_list,
                         index=opt_index,
                         key="DB_BANDER",
                         on_change=on_change_db_vender)
            session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DB_BANDER", session.DB_BANDER)

            db_vensor = session["DPM_SESSION"]["DB_BANDER"]
            if db_vensor == "MariaDB" or db_vensor == "MySQL":
                url_db_type = "mysql"
            elif db_vensor == "PostgreSQL":
                url_db_type = "sql"

        bound_area = session["DPM_SESSION"]["SYSTEM_AREA"]
        if session["DPM_SESSION"]["REGISTER_OPTION"] == "등록":
            host_address = st.text_input("Host Address", "")
            host_port = st.text_input("Host Port", "")
            db_name = st.text_input("데이터베이스 이름", "")
            username = st.text_input("User Name", "")
            password = st.text_input("Password", type="password")
        elif session["DPM_SESSION"]["REGISTER_OPTION"] == "수정":
            system_name = session["DPM_SESSION"]["EDIT_SYSTEM_NAME"]
            edit_db_name = session["DPM_SESSION"]["EDIT_DB_NAME"]
            sel_row = etl_inoutbound_df.loc[
                (etl_inoutbound_df["시스템 이름"] == system_name) & (etl_inoutbound_df["데이터베이스 이름"] == edit_db_name)]
            st.write("데이터베이스 이름")
            st.write("Host Address")
            host_address = sel_row["Host Address"].item()
            st.code(host_address, language="markdown")
            st.write("Host Port")
            host_port = sel_row["Host Port"].item()
            st.code(host_port, language="markdown")
            st.write("데이터베이스 이름")
            db_name = sel_row["데이터베이스 이름"].item()
            st.code(db_name, language="markdown")
            username = st.text_input("User Name", sel_row["User Name"].item())
            password = st.text_input("Password", sel_row["Password"].item(), type="password")
        elif session["DPM_SESSION"]["REGISTER_OPTION"] == "삭제":
            drop_system_name = session["DPM_SESSION"]["DROP_SYSTEM_NAME"]
            if etl_inoutbound_df is not None:
                doc_filter = {'db_name': session["DPM_SESSION"]["DROP_DB_NAME"]}
                collection_name = f"etl_{bound_area}_RDB_{drop_system_name}"
                submitted = st.button('{} 삭제'.format(session["DPM_SESSION"]["DROP_DB_NAME"]))
                if submitted:
                    drop_doc_mongo(collection_name, doc_filter)
                    st.rerun()

        if session["DPM_SESSION"]["REGISTER_OPTION"] == "등록":
            submitted = st.button("등록")
            if submitted:
                if len(password) > 0:
                    if db_vensor in option_db_vender_list:
                        collection_nm_type = "RDB"
                    bucket_filter = {'sys_name': system_name}
                    object_filter = {'db_name': db_name}
                    if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}",
                                         filter=bucket_filter):
                        if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                             filter=object_filter):
                            st.info(f"시스템 이름 {system_name}과 데이터베이스 {db_name}이 존재합니다.")
                        else:
                            object_filter = {'doc_key': 'db_name', 'doc_value': db_name}
                            system_dbs_document = {'db_name': db_name,
                                                   'username': username,
                                                   'password': password,
                                                   "db_type": url_db_type}
                            upsert_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                             filter=object_filter,
                                             document=system_dbs_document)
                    else:
                        up_document = {'bound_area': bound_area, 'sys_name': system_name, 'host_address': host_address,
                                       'host_port': host_port}
                        upsert_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}", filter=bucket_filter,
                                         document=up_document)
                        object_filter = {'doc_key': 'db_name', 'doc_value': db_name}
                        system_dbs_document = {'db_name': db_name, 'username': username, 'password': password,
                                               "db_type": url_db_type}
                        upsert_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                         filter=object_filter,
                                         document=system_dbs_document)
                    if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}",
                                         filter=bucket_filter):
                        if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                             filter=object_filter):
                            st.write("등록 성공")
                            st.rerun()
                else:
                    st.error("암호가 비어 있습니다.")
        elif session["DPM_SESSION"]["REGISTER_OPTION"] == "수정":
            submitted = st.button("수정")
            if submitted:
                if len(password) > 0:
                    collection_nm_type = "RDB"
                    system_doc = {'bound_area': bound_area, 'sys_name': system_name, 'host_address': host_address,
                                  'host_port': host_port}
                    bucket_filter = {'doc_key': 'sys_name', 'doc_value': system_name}
                    upsert_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}",
                                     filter=bucket_filter,
                                     document=system_doc)

                    object_filter = {'doc_key': 'db_name', 'doc_value': db_name}
                    system_dbs_document = {'db_name': db_name, 'username': username, 'password': password,
                                           "db_type": url_db_type}
                    upsert_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                     filter=object_filter,
                                     document=system_dbs_document)

                if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}",
                                     filter=bucket_filter):
                    if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{collection_nm_type}_{system_name}",
                                         filter=object_filter):
                        st.write("수정 성공")
                        st.rerun()


def main():
    dpm_ss = redis_mgr.get("DPM_SESSION")
    if dpm_ss is not None:
        elk_logger.debug({sys._getframe().f_code.co_name:f"DPM:session realod"})
        session["DPM_SESSION"] = dict(json.loads(dpm_ss.decode('utf-8')))
    else:
        session["DPM_SESSION"] = None
        elk_logger.debug({sys._getframe().f_code.co_name:f"DPM:None session"})
    st.title("Data Manager")

    with st.sidebar:
        st.markdown("### Machine Learning Data Manager")
        option_main_nume_list =["저장소 등록","연계"]
        opt_index = get_IndexofWidget(session["DPM_SESSION"], option_main_nume_list, "MAIN_MENU")
        st.selectbox(label="데이터 저장소",
                     options=option_main_nume_list,
                     index=opt_index,
                     key="MAIN_MENU",
                     on_change=on_change_main_menu)
        session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MAIN_MENU", session.MAIN_MENU)

        if session["DPM_SESSION"]["MAIN_MENU"] == "저장소 등록":            # 저장소 등록 선택 시 나오는 추가 메뉴
            option_bound_list = ["InBound", "OutBound", "AWS", "BigDataEngine"]
            opt_index = get_IndexofWidget(session["DPM_SESSION"], option_bound_list, "SYSTEM_AREA")
            st.selectbox(label="데이터 채널",
                         options=option_bound_list,
                         index=opt_index,
                         key="SYSTEM_AREA",
                         on_change=on_change_system_area)
            session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "SYSTEM_AREA", session.SYSTEM_AREA)

            if session["DPM_SESSION"]["SYSTEM_AREA"] == "InBound":
                option_storagetype_list = ["관계형DB"]
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_storagetype_list, "STORAGE_TYPE")
                st.selectbox(label='{} 데이터 저장소 타입'.format(session["DPM_SESSION"]["SYSTEM_AREA"]),
                             options=option_storagetype_list,
                             index=opt_index,
                             key="STORAGE_TYPE",
                             on_change=on_change_storagetype)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "STORAGE_TYPE", session.STORAGE_TYPE)

            elif session["DPM_SESSION"]["SYSTEM_AREA"] == "OutBound":
                option_storagetype_list = ["관계형DB"]
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_storagetype_list, "STORAGE_TYPE")
                st.selectbox(label='{} 데이터 저장소 타입'.format(session["DPM_SESSION"]["SYSTEM_AREA"]),
                             options=option_storagetype_list,
                             index=opt_index,
                             key="STORAGE_TYPE",
                             on_change=on_change_storagetype)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "STORAGE_TYPE", session.STORAGE_TYPE)

            elif session["DPM_SESSION"]["SYSTEM_AREA"] == "AWS":
                option_storagetype_list = ["S3"]
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_storagetype_list, "AWS_SVC_TYPE")
                st.selectbox(label="데이터 저장소 타입",
                             options=option_storagetype_list,
                             index=opt_index,
                             key="AWS_SVC_TYPE",
                             on_change=on_change_aws_svc_type)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_SVC_TYPE", session.AWS_SVC_TYPE)

            elif session["DPM_SESSION"]["SYSTEM_AREA"] == "BigDataEngine":
                option_engline_list = ["Trino"]
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_engline_list, "BIGDATAENGINE")
                st.selectbox(label="쿼리 엔진",
                             options=option_engline_list,
                             index=opt_index,
                             key="BIGDATAENGINE",
                             on_change=on_change_bigdateengine_type)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "BIGDATAENGINE", session.BIGDATAENGINE)
    ####
    # out of sidebar
    ####
    if session["DPM_SESSION"]["MAIN_MENU"] == "저장소 등록":
        if session["DPM_SESSION"]["SYSTEM_AREA"] == "InBound" or session["DPM_SESSION"]["SYSTEM_AREA"] == "OutBound":
            regist_inout_rdb()

        elif session["DPM_SESSION"]["SYSTEM_AREA"] == "BigDataEngine":
            etl_catalog_list = get_etl_trino_catalogs_mongo()
            if len(etl_catalog_list) > 0:
                etl_catalog_df = pd.DataFrame(etl_catalog_list)
                st.dataframe(etl_catalog_df, width=700, height=300)
            else:
                st.info("등록된 저장소가 없습니다.")
            trino_ip = st.text_input("Host IP", "")
            trino_port = st.text_input("Host Port", "")
            trino_user = st.text_input("사용자 ID", "")
            if len(trino_ip) > 0 and len(trino_port) > 0 and len(trino_user) > 0:
                t_conn = trino_conn(host=trino_ip,port=trino_port,  user=trino_user )
                cur = t_conn.cursor()
                cur.execute("SHOW catalogs")
                catalogs = cur.fetchall()
                catalog_list = []
                for catalog in catalogs:
                    catalog_list.append(catalog[0])
                opt_index = get_IndexofWidget(session["DPM_SESSION"], catalog_list, "TRINO_CATALOGS")
                st.selectbox(label="Trino 카탈로그",
                             options=catalog_list,
                             index=opt_index,
                             key="TRINO_CATALOGS",
                             on_change=on_change_catalog)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "TRINO_CATALOGS",
                                                                session.TRINO_CATALOGS)
                submitted = st.button("등록")
                if submitted:
                    bound_area = "Trino"
                    if len(trino_user) > 0:
                        catalog_name = session["DPM_SESSION"]["TRINO_CATALOGS"]
                        catalog_doc_filter = {'catalog': catalog_name}
                        if isexist_mongo_doc(collection_name=f"etl_{bound_area}",
                                             filter=catalog_doc_filter):
                            st.info(f"카탈로그 이름 {catalog_name}이 존재합니다.")

                        else:
                            trino_document = {'trino_ip': trino_ip, 'trino_port': trino_port,'trino_user': trino_user}
                            upsert_mongo_doc(collection_name=f"etl_{bound_area}",
                                             filter=catalog_doc_filter,
                                             document=trino_document)
                            if isexist_mongo_doc(collection_name=f"etl_{bound_area}",
                                                 filter=catalog_doc_filter):
                                    st.write("등록 성공")
                                    st.rerun()
                    else:
                        st.error("사용자 이름이 비어 있습니다.")
        elif session["DPM_SESSION"]["SYSTEM_AREA"] == "AWS":
            if session["DPM_SESSION"]["AWS_SVC_TYPE"] == "S3":
                # etl_aws_s3_buckets = get_etl_aws_s3_details_list()
                etl_aws_s3_buckets = get_etl_aws_s3_bucket_list()
                if len(etl_aws_s3_buckets) > 0:
                    etl_object_df = pd.DataFrame(etl_aws_s3_buckets)
                    st.dataframe(etl_object_df, height=300)
                else:
                    etl_object_df = None
                    st.info("등록된 저장소가 없습니다.")
                if etl_object_df is None:
                    register_option = ["등록"] #S3 접근 정보 등록 후 버킷, 객체 자동 등록
                else:
                    register_option = ["등록","삭제"] #S3 접근 정보만 수정 및 삭제
                opt_index = get_IndexofWidget(session["DPM_SESSION"], register_option, "REGISTER_OPTION")
                st.selectbox(label="등록/삭제",
                             options=register_option,
                             index=opt_index,
                             key="REGISTER_OPTION",
                             on_change=on_change_register_option)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "REGISTER_OPTION", session.REGISTER_OPTION)

                if session["DPM_SESSION"]["REGISTER_OPTION"] == "등록":
                    service_name = st.text_input("시스템 이름", "S3")
                    bound_area = session["DPM_SESSION"]["SYSTEM_AREA"]
                    region_list = ["us-east-2",
                    "us-east-1",
                    "us-west-1",
                    "us-west-2",
                    "af-south-1",
                    "ap-east-1",
                    "ap-south-2",
                    "ap-southeast-3",
                    "ap-southeast-4",
                    "ap-south-1",
                    "ap-northeast-3",
                    "ap-northeast-2",
                    "ap-southeast-1",
                    "ap-southeast-2",
                    "ap-northeast-1",
                    "ca-central-1",
                    "ca-west-1",
                    "eu-central-1",
                    "eu-west-1",
                    "eu-west-2",
                    "eu-south-1",
                    "eu-west-3",
                    "eu-south-2",
                    "eu-north-1",
                    "eu-central-2",
                    "il-central-1",
                    "me-south-1",
                    "me-central-1",
                    "sa-east-1"]
                    aws_region = st.selectbox(label="리전", options=region_list, index=11)
                    access_key = st.text_input("accesskey", "type your access key")
                    secret_key = st.text_input("secretkey","type your secret key")

                    bucket_list = get_aws_s3_bucket_list(access_key, secret_key, aws_region)
                    print(type(bucket_list))
                    if isinstance(bucket_list, str):
                        st.error(bucket_list)
                    elif isinstance(bucket_list, list):
                        bucket_infos = []
                        for bucket_name in bucket_list:
                            object_list = get_aws_s3_object_list(access_key, secret_key, aws_region, bucket_name)
                            for file_name in object_list:
                                bucket_infos.append({"리전": aws_region, "버킷이름": bucket_name, "파일 명": file_name})
                        df = pd.DataFrame(bucket_infos)
                        st.dataframe(df)

                        opt_index = get_IndexofWidget(session["DPM_SESSION"], bucket_list, "BUCKET_NAME")
                        st.selectbox(label="등록할 버켓 이름",
                                     options=bucket_list,
                                     index=opt_index,
                                     key="BUCKET_NAME",
                                     on_change=on_change_bucket_option)

                        session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "BUCKET_NAME",
                                                                        session.BUCKET_NAME)
                        bucket_name = session["DPM_SESSION"]["BUCKET_NAME"]
                    else:
                        show_ClientError_detail=st.toggle('상세 보기')
                        if show_ClientError_detail:
                            st.error(bucket_list)

                    submitted = st.button("등록")
                    if submitted:
                        bucket_filter = {'bucket_name': bucket_name}
                        if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{service_name}", filter=bucket_filter):
                            st.write("이미 버킷이 존재합니다.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            bucket_doc = {'aws_region': aws_region, 'access_key': access_key, 'secret_key': secret_key}
                            upsert_mongo_doc(collection_name=f"etl_{bound_area}_{service_name}", filter=bucket_filter,  document=bucket_doc)
                            if isexist_mongo_doc(collection_name=f"etl_{bound_area}_{service_name}",  filter=bucket_filter):
                                st.write("등록 성공")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.write("등록 오류")
                # elif session["DPM_SESSION"]["REGISTER_OPTION"] == "수정":
                #     if etl_object_df is not None:
                #         bucket_list = etl_object_df["버킷 이름"].tolist()
                #         opt_index = get_IndexofWidget(session["DPM_SESSION"], bucket_list, "EDIT_SYSTEM_NAME")
                #         st.selectbox(label="수정할 버킷을 선택하세요",
                #                      options=bucket_list,
                #                      index=opt_index,
                #                      key="EDIT_SYSTEM_NAME",
                #                      on_change=on_change_edit_systemname)
                #         session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "EDIT_SYSTEM_NAME",
                #                                                         session.EDIT_SYSTEM_NAME)
                elif session["DPM_SESSION"]["REGISTER_OPTION"] == "삭제":
                    if etl_object_df is not None:
                        system_list = etl_object_df["버킷 이름"].tolist()
                        opt_index = get_IndexofWidget(session["DPM_SESSION"], system_list, "DROP_BUCKET_NAME")
                        st.selectbox(label="버킷을 선택하세요",
                                     options=system_list,
                                     index=opt_index,
                                     key="DROP_BUCKET_NAME",
                                     on_change=on_change_drop_bucketname)
                        session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "DROP_BUCKET_NAME",
                                                                        session.DROP_BUCKET_NAME)
                        drop_bucket_name = session["DPM_SESSION"]["DROP_BUCKET_NAME"]
                        filter = {"bucket_name": drop_bucket_name}
                        submitted = st.button("삭제")
                        if submitted:
                            if drop_doc_mongo(collection_name=f"etl_AWS_S3", drop_filter=filter):
                                st.info("삭제 성공")
                            else:
                                st.error("삭제 오류")


    #---------데이터 이동
    elif session["DPM_SESSION"]["MAIN_MENU"] == "연계":
        option_main_nume_list = ['InBound 마이그레이션', 'OutBound 마이그레이션', 'AWS 마이그레이션', "BigDataEngine"]
        opt_index = get_IndexofWidget(session["DPM_SESSION"], option_main_nume_list, "MIGRATION_CHANNEL")
        st.selectbox(label="연계 채널 선택",
                      options=option_main_nume_list,
                      index=opt_index,
                      key="MIGRATION_CHANNEL",
                      on_change=on_change_migraion_path)
        session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "MIGRATION_CHANNEL", session.MIGRATION_CHANNEL)

        if session["DPM_SESSION"]["MIGRATION_CHANNEL"] == "InBound 마이그레이션":
            option_inb_mig_list = keys_sysnameinb_from_mongo()
            if len(option_inb_mig_list) > 0:
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_inb_mig_list, "INB_MIGRATION_CHANNEL")
                st.selectbox(label="In Bound 시스템 선택",
                             options=option_inb_mig_list,
                             index=opt_index,
                             key="INB_MIGRATION_CHANNEL",
                             on_change=on_change_inb_migraion_path)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "INB_MIGRATION_CHANNEL",
                                                                session.INB_MIGRATION_CHANNEL)
                analysis_migraion_inoutbound_data('In', session["DPM_SESSION"]["INB_MIGRATION_CHANNEL"])
            else:
                st.info("연계할 등록된 저장소가 없습니다.")

        elif session["DPM_SESSION"]["MIGRATION_CHANNEL"] == "OutBound 마이그레이션":
            option_outb_mig_list = keys_sysnameoutb_from_mongo()
            if len(option_outb_mig_list) > 0:
                opt_index = get_IndexofWidget(session["DPM_SESSION"], option_outb_mig_list, "OUTB_MIGRATION_CHANNEL")
                st.selectbox(label="Out Bound 시스템 선택",
                             options=option_outb_mig_list,
                             index=opt_index,
                             key="OUTB_MIGRATION_CHANNEL",
                             on_change=on_change_outb_migraion_path)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "OUTB_MIGRATION_CHANNEL",
                                                                session.OUTB_MIGRATION_CHANNEL)
                analysis_migraion_inoutbound_data('Out', session["DPM_SESSION"]["OUTB_MIGRATION_CHANNEL"])
            else:
                st.info("연계할 등록된 저장소가 없습니다.")

        elif session["DPM_SESSION"]["MIGRATION_CHANNEL"] == "AWS 마이그레이션":
            option_aws_mig_list = ["S3"]
            opt_index = get_IndexofWidget(session["DPM_SESSION"], option_aws_mig_list, "AWS_SVC")
            st.selectbox(label="AWS 서비스",
                         options=option_aws_mig_list,
                         index=opt_index,
                         key="AWS_SVC",
                         on_change=on_change_aws_svc_info)
            session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_SVC",
                                                            session.AWS_SVC)

            buckets_info = get_etl_aws_s3_bucket_list()
            bucket_name = [bucket_info.get('버킷 이름') for bucket_info in  buckets_info]
            if len(bucket_name) > 0:
                opt_index = get_IndexofWidget(session["DPM_SESSION"], bucket_name, "AWS_BUCKET")
                st.selectbox(label="AWS Bucket",
                             options=bucket_name,
                             index=opt_index,
                             key="AWS_BUCKET",
                             on_change=on_change_aws_bucket_info)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_BUCKET",
                                                                session.AWS_BUCKET)

            s3_bucket_object_infos = get_aws_s3_all_infos(session["DPM_SESSION"]["AWS_BUCKET"])
            if len(s3_bucket_object_infos) > 0:
                object_list = [ s3_bucket_object_info["Object Name"] for s3_bucket_object_info in  s3_bucket_object_infos]
                opt_index = get_IndexofWidget(session["DPM_SESSION"], object_list, "AWS_S3_OBJECT")
                st.selectbox(label="파일",
                             options=object_list,
                             index=opt_index,
                             key="AWS_S3_OBJECT",
                             on_change=on_change_aws_object_info)
                session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "AWS_S3_OBJECT",
                                                                session.AWS_S3_OBJECT)

                bucket_name = session["DPM_SESSION"]["AWS_BUCKET"]
                file_name = session["DPM_SESSION"]["AWS_S3_OBJECT"]
                bucket_access_info = get_document_mongo("etl_AWS_S3", "bucket_name", bucket_name)
                region = bucket_access_info["aws_region"]
                access_key = bucket_access_info["access_key"]
                secret_key = bucket_access_info["secret_key"]
                bucket_name = bucket_access_info["bucket_name"]
                update_aws_s3_bucket_object_info(region, access_key, secret_key, bucket_name, file_name)
            else:
                st.info("연계할 등록된 저장소가 없습니다.")

        elif session["DPM_SESSION"]["MIGRATION_CHANNEL"] == "BigDataEngine":
            st.write("Trino Schema 목록")
            link_trino_catalogs = get_etl_trino_catalogs_mongo()
            schemas_infos = []
            for link_trino_catalog in link_trino_catalogs:
                trino_ip = link_trino_catalog['Host Address']
                trino_port = link_trino_catalog['Host Port']
                user_name = link_trino_catalog['User Name']
                catalog_nm = link_trino_catalog['catalog']
                schemas = get_schemas_in_catalog_dict(trino_ip, trino_port, user_name, catalog_nm)['Schema']
                for schema in schemas:
                    schema_info = {
                        'Trino Address':trino_ip,
                        'Trino Port':trino_port,
                        'User Name':user_name,
                        'catalog':catalog_nm,
                        'schema': schema
                    }
                    schemas_infos.append(schema_info)
            schemas_info_df = pd.DataFrame(schemas_infos)
            st.dataframe(schemas_info_df, width=700, hide_index=True)

            st.write("Trino 테이블 목록")
            for idx in range(schemas_info_df.shape[0]):
                schema_row = schemas_info_df.loc[[idx]]
                trino_address = schema_row['Trino Address'].item()
                trino_port = schema_row['Trino Port'].item()
                user_name = schema_row['User Name'].item()
                catalog = schema_row['catalog'].item()
                schema = schema_row['schema'].item()
                table_info_dict = get_table_in_trino_schema_dict(trino_address, trino_port, user_name, catalog, schema)
                table_infos=[]
                for table_name in table_info_dict['Table']:
                    schema_table_info = {
                        "Schema":schema,
                        "table": table_name
                    }
                    table_infos.append(schema_table_info)
                table_infos_df = pd.DataFrame(table_infos)
                st.dataframe(table_infos_df, width=700, hide_index=True)

            trino_sql = st.text_area('Trino SQL Editor', value="SELECT * FROM ")
            if 'TRINO_SQL_DF' not in session:
                session["TRINO_SQL_DF"] = pd.DataFrame()
            if st.button("Query Run"):
                if len(trino_sql) > 0:
                    sql_result_df = run_trino_sql(trino_address, trino_port, user_name, trino_sql)
                    if sql_result_df.shape[0] > 0:
                        session["TRINO_SQL_DF"] = sql_result_df
                        st.dataframe(sql_result_df)

            if session["TRINO_SQL_DF"].shape[0] > 0:
                object_name = st.text_input("객체 이름", value='diabetes.csv')
                if st.button("복사"):
                    update_object(session["TRINO_SQL_DF"], object_name)
                    if is_fileobj_inbucket(DS_BUCKET_NAME, object_name) == True:
                        st.write(f"ML Storage에 {object_name} 생성 완료")



        elif session["DPM_SESSION"]["MIGRATION_CHANNEL"] == "GCP 마이그레이션":
            option_aws_mig_list = ["S3"]
            opt_index = get_IndexofWidget(session["DPM_SESSION"], option_aws_mig_list, "GCP_MIGRATION_CHANNEL")
            # st.selectbox(label="AWS 마이그레이션",
            #              options=option_aws_mig_list,
            #              index=opt_index,
            #              key="GCP_MIGRATION_CHANNEL",
            #              on_change=on_change_aws_migraion_path)
            session["DPM_SESSION"] = upsert_sessiondb_redis(session["DPM_SESSION"], "GCP_MIGRATION_CHANNEL",
                                                            session.AWS_MIGRATION_CHANNEL)



if __name__ == "__main__":
    st.set_page_config(
        page_title="Machine Learning Data Manager", page_icon=":chart_with_upwards_trend:"
    )
    main()
