import io
import json
import os
import pprint
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit import session_state as session
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
import pika
from sklearn.preprocessing import MinMaxScaler

from StorageManage.redis import redis_mgr
from dilogger import DIlogger
from multilanguage.multi_lang import item_caption
from sklearn.decomposition import PCA
from pca import pca
from st_keyup import st_keyup
from pymongo import MongoClient
import socket

from post_process_algo.randomforest_regression import do_rfr_report, get_rfr_model

st.set_option('deprecation.showPyplotGlobalUse', False)
log_handler = os.environ.get('log_handler')

mqbroker = os.environ.get('mqbroker')
redis_host = os.environ.get('redis_host')

from post_process_algo.randomforest_classifier import do_rfc_report, get_rfc_model
from post_process_algo.statistics import do_statistics_report
from StorageManage.minio import get_files_minio, load_dataset_minio, minio_client, DS_BUCKET_NAME, drop_dataset

import logging

basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(basedir)

elk_address = os.environ.get('elk_address')

if elk_address is not None:
    address_list = [elk_address]
else:
    address_list = None

index = 'dataignite-boltzman-log'
dilogger = DIlogger(address_list = address_list, index = index, level="DEBUG")
dilogger.info({sys._getframe().f_code.co_name:f"Host name:{socket.gethostname()}"})
def get_IndexofWidget(DIS_SESSION, values, widget_type):
    if DIS_SESSION is None:
        return 0
    else:
        if widget_type in DIS_SESSION:
            session_widget_value = DIS_SESSION[widget_type]
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

def get_ValuesofWidget(DIS_SESSION, column_list, widget_type):
    if DIS_SESSION is None:
        return column_list
    else:
        if widget_type in DIS_SESSION:
            session_widget_value = DIS_SESSION[widget_type]
            if session_widget_value is None:
                return column_list
            else:
                return session_widget_value
        return column_list

def upsert_session(DIS_SESSION, key, value):
    if DIS_SESSION is None:
        DIS_SESSION = {key: value}
    else:
        DIS_SESSION[key] = value
    json_dpm_session = json.dumps(DIS_SESSION, ensure_ascii=False).encode('utf-8')
    redis_mgr.set("DIS_SESSION", json_dpm_session)
    # redis_mgr.expire("DIS_SESSION", 60) #TTL 1min
    return DIS_SESSION

dpm_ss = redis_mgr.get("DIS_SESSION")
if dpm_ss is not None:
    session["DIS_SESSION"] = dict(json.loads(dpm_ss.decode('utf-8')))
else:
    session["DIS_SESSION"] = None


class WebMain():
    def on_change_train_ds(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "TRAIN_DS", session.TRAIN_DS)

    def on_change_algo_category(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ALGORITHM_CATEGORY", session.ALGORITHM_CATEGORY)

    def on_change_algorithm_statistics(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_STATISTICS", session.ACT_STATISTICS)

    def on_change_algorithm_classification(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_ALGORITHM_CLASSIFICATION", session.ACT_ALGORITHM_CLASSIFICATION)

    def on_change_algorithm_regression(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_ALGORITHM_REGRESSION",
                                                session.ACT_ALGORITHM_REGRESSION)

    def on_change_classification_input(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "CLASSIFICATION_INPUT",
                                                session.CLASSIFICATION_INPUT)

    def on_change_classification_output(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "CLASSIFICATION_OUTPUT",
                                                session.CLASSIFICATION_OUTPUT)

    def on_change_regression_input(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "REGRESSION_INPUT",
                                                session.REGRESSION_INPUT)

    def on_change_regression_output(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "REGRESSION_OUTPUT",
                                                session.REGRESSION_OUTPUT)
    def on_change_statistics_descriptive_input(self):
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "STATISTICS_DESC_INPUT",
                                                session.STATISTICS_DESC_INPUT)

    def __init__(self):
        credentials = pika.PlainCredentials('guest', 'guest')
        if mqbroker == None:
            dilogger.debug({sys._getframe().f_code.co_name:"rabbitmq host:localhsot"})
            self.pika_conn_param = pika.ConnectionParameters('localhost', 5672, '/', credentials)
        else:
            dilogger.debug({sys._getframe().f_code.co_name: f"rabbitmq host:{mqbroker}"})
            self.pika_conn_param = pika.ConnectionParameters(mqbroker, 5672, '/', credentials)

        self.ALGO_OPTIONS = ['Statistics', 'Classification', 'Regression']

        self.ML_MODULE_LIST = ['Classification', 'Regression', 'Clustering']

        self.CLASSIFICATION_OPTIONS = ['Randomforest Classifier']
        self.REGRESSION_OPTIONS = ['Randomforest Regression']
        self.STATISTICS_OPTIONS = ['Descriptive']

        self.MODEL_RUN_OPT = ['Create new model', 'Review Model']
        self.init_session()
        session['list_dataset'] = get_files_minio(DS_BUCKET_NAME)

        self.preprocessed_data = None
        self.input_field = None
        self.class_field = None
        self.get_MongoMgr()
    def get_MongoMgr(self):
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
        mongo_client = MongoClient(host=mongo_host,
                                   username=mongo_user,
                                   password=mongo_passwd,
                                   authSource=auth_source,
                                   authMechanism='SCRAM-SHA-256')

        self.mongo_db = mongo_client.get_database(auth_source)
        if self.mongo_db.name != auth_source:
            st.error(f"MongoDB Connect failed")


    def init_session(self):
        dilogger.debug({sys._getframe().f_code.co_name: ""})
        session['LANG'] = 'kor'
        if 'ALGO_CATEGORY' not in session: session['ALGO_CATEGORY'] = None
        if 'ACT_ALGORITHM' not in session: session['ACT_ALGORITHM'] = None
        if 'BTN_RPT_DISABLE' not in session: session['BTN_RPT_DISABLE'] = True
        if 'Preprocess' not in session:
            session['Preprocess'] = {'session_id': None,
                                     'session_name': None,
                                     'status': None,
                                     'Report_enable': False}

        if 'Statistics' not in session:
            session['Statistics'] = {'session_id': None,
                                     'session_name': None,
                                     'status': None,
                                     'Report_enable': False}

        if 'Randomforest Classifier' not in session:
            session['Randomforest Classifier'] = {'session_id': None,
                                                  'session_name': None,
                                                  'status': None,
                                                  'Report_enable': False}

        if 'ml_run_mode' not in session: session['ml_run_mode'] = None
        if 'prev_algo_name' not in session: session['prev_algo_name'] = None
        if 'preprocess_df' not in session: session['preprocess_df'] = None
        if 'update_rename_ds' not in session: session['update_rename_ds'] = None
        if 'update_ds_rtn' not in session:  session['update_ds_rtn'] = None
        if 'btn_fit_disable' not in session: session['btn_fit_disable'] = True
        if 'drop_file_status' not in session: session.drop_file_status = False
        if 'list_tables' not in session: session['list_tables'] = None
        if 'sel_train_ds' not in session: session['sel_train_ds'] = None

    def view_history(self):
        dilogger.debug({sys._getframe().f_code.co_name: ""})
        model_keys = redis_mgr.keys()
        model_list = []
        model_id_list = []
        st.write("---")
        st.write(f"모델 목록", unsafe_allow_html=True)
        algo_type = session["DIS_SESSION"]['ACT_ALGORITHM']
        key_prefix = None
        if algo_type == 'Randomforest Classifier':
            key_prefix = 'rfc'
        elif algo_type == 'Randomforest Regression':
            key_prefix = 'rfr'
        elif algo_type == 'Statistics':
            key_prefix = 'statistics'

        for model_key in model_keys:
            if key_prefix in str(model_key):
                 if 'report' not in model_key.decode('utf-8'):
                    model_status = redis_mgr.get(model_key).decode('utf-8')
                    ms_json = json.loads(model_status)
                    model_list.append(ms_json)
                    model_id_list.append(ms_json['session_id'])

        model_history_df = pd.DataFrame.from_dict(model_list)
        st.dataframe(model_history_df)
        st.write("---")
        # use custom dataset
        custom_file = st.file_uploader(f"#### 모델 테스트 파일",
                                     type=['xls', 'csv'],
                                     key='rfc_view_history')
        custom_df = self.get_file_df(custom_file)
        st.write("---")
        model_ids = st.multiselect(f"모델 선택", model_id_list)
        for model_id in model_ids:
            if key_prefix == 'rfc':
                do_rfc_report(dilogger, model_id, flexible=True, custom_df=custom_df)
            elif key_prefix == 'rfr':
                do_rfr_report(dilogger, model_id, flexible=True, custom_df=custom_df)


    def on_click_drop_file(self):
        if session.txtinput_filename == session.sel_object_ds:
            dilogger.debug({sys._getframe().f_code.co_name: f"DELETE {session.txtinput_filename}"})
            if drop_dataset(DS_BUCKET_NAME, session.sel_object_ds) == False:
                st.error('파일 제거 오류 발생')

    def get_file_df(self, custom_file):
        dilogger.debug({sys._getframe().f_code.co_name: ""})
        custom_df = None
        if custom_file is not None:
            file_name = custom_file.name
            bytes_data = custom_file.getvalue()
            _, file_type = os.path.splitext(file_name)
            if 'xls' in file_type:
                custom_df = pd.read_excel(bytes_data)
                st.dataframe(custom_df)
                dilogger.debug({sys._getframe().f_code.co_name:f"{file_name} loaded"})

            elif 'csv'  in file_type:
                object_str = str(bytes_data, 'utf-8')
                data = io.StringIO(object_str)
                custom_df = pd.read_csv(data)
                st.write(f"파일 미리보기")
                st.dataframe(custom_df)
                dilogger.debug({sys._getframe().f_code.co_name:f"{file_name} loaded"})
            # elif 'png' or 'jpg' in file_type:
            #     st.image(file_name, caption=f"{file_name}")
            #     dilogger.debug({sys._getframe().f_code.co_name:f"{file_name} loaded"})
            else:
                st.error(f"처리 할 수 없는 데이터 입니다.")
                st.stop()
        return custom_df

    def configure_menu(self):
        dilogger.debug({sys._getframe().f_code.co_name: ""})
        st.markdown("---")
        st.markdown(f"### {item_caption['data_manage'][session['LANG']]}")
        self.uploaded_file = st.file_uploader(f"#### {item_caption['file_upload'][session['LANG']]}",
                                              type=['xls', 'csv'])
        if self.uploaded_file is not None:
            file_name = self.uploaded_file.name
            bytes_data = self.uploaded_file.getvalue()
            bio = io.BytesIO(bytes_data)
            file_size = len(bytes_data)
            minio_client.put_object(DS_BUCKET_NAME, file_name, bio, file_size)
            dilogger.debug({sys._getframe().f_code.co_name:f"Upload {self.uploaded_file.name} finished"})
            st.text(f'Upload {self.uploaded_file.name} finished')
            session['list_dataset'] = get_files_minio(DS_BUCKET_NAME)
        st.markdown("---")

        opt_index = get_IndexofWidget(session["DIS_SESSION"], session['list_dataset'], "TRAIN_DS")
        st.selectbox(
            label=f"#### {item_caption['listup_ds'][session['LANG']]}",
            options=session['list_dataset'],
            index=opt_index,
            key="TRAIN_DS",
            on_change=self.on_change_train_ds)
        session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "TRAIN_DS", session.TRAIN_DS)

        dilogger.debug({sys._getframe().f_code.co_name: f"{session['DIS_SESSION']['TRAIN_DS']} SELECTED"})
        if st.button(label=f"{session["DIS_SESSION"]['TRAIN_DS']} 삭제"):
            session.drop_file_status = False
            st.warning(f"""파일을 삭제하면 복구할 수 없습니다. 그래도 삭제하려면 파일명 {session['sel_train_ds']}을 기입하세요""")
            st.text_input("파일명", on_change=self.get_file_name, key='txtinput_filename')
            st.button(f"파일 삭제", on_click=self.on_click_drop_file)
        st.markdown("---")
        session['ml_run_mode'] = None
        if session["DIS_SESSION"]['TRAIN_DS'] is not None:
            opt_index = get_IndexofWidget(session["DIS_SESSION"], self.ALGO_OPTIONS, "ALGORITHM_CATEGORY")
            st.selectbox(
                label=f"#### {item_caption['select_algorithm'][session['LANG']]}",
                options=self.ALGO_OPTIONS,
                index=opt_index,
                key='ALGORITHM_CATEGORY',
                on_change=self.on_change_algo_category)
            session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ALGORITHM_CATEGORY", session.ALGORITHM_CATEGORY)

            if session["DIS_SESSION"]['ALGORITHM_CATEGORY'] == 'Statistics':
                opt_index = get_IndexofWidget(session["DIS_SESSION"], self.STATISTICS_OPTIONS,"ACT_ALGORITHM_STATISTICS")
                st.selectbox(
                    label=f"#### {'통계 옵션'}",
                    options=self.STATISTICS_OPTIONS,
                    index=opt_index, key='ACT_ALGORITHM_STATISTICS',
                    on_change=self.on_change_algorithm_statistics)
                session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_ALGORITHM", session.ACT_ALGORITHM_STATISTICS)

            elif session["DIS_SESSION"]['ALGORITHM_CATEGORY'] == 'Classification':
                opt_index = get_IndexofWidget(session["DIS_SESSION"], self.CLASSIFICATION_OPTIONS, "ACT_ALGORITHM_CLASSIFICATION")
                st.selectbox(
                    label=f"#### {item_caption['msg_sel_classification_algorithm'][session['LANG']]}",
                    options=self.CLASSIFICATION_OPTIONS,
                    index=opt_index, key='ACT_ALGORITHM_CLASSIFICATION',
                    on_change=self.on_change_algorithm_classification)
                session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_ALGORITHM", session.ACT_ALGORITHM_CLASSIFICATION)

            elif session["DIS_SESSION"]['ALGORITHM_CATEGORY'] == 'Regression':
                optindex = get_IndexofWidget(session["DIS_SESSION"], self.REGRESSION_OPTIONS,"ACT_ALGORITHM_REGRESSION")
                st.selectbox(
                    label=f"#### {item_caption['msg_sel_regression_algorithm'][session['LANG']]}",
                    options=self.REGRESSION_OPTIONS,
                    index=optindex, key='ACT_ALGORITHM_REGRESSION',
                    on_change=self.on_change_algorithm_regression)
                session["DIS_SESSION"] = upsert_session(session["DIS_SESSION"], "ACT_ALGORITHM", session.ACT_ALGORITHM_REGRESSION)

                dilogger.debug({sys._getframe().f_code.co_name: f"{session['ACT_ALGORITHM']} SELECTED"})
            elif session["DIS_SESSION"]['ALGORITHM_CATEGORY'] == 'Clustering':
                pass

            if session["DIS_SESSION"]['ALGORITHM_CATEGORY'] is not None:
                if session['prev_algo_name'] != session["DIS_SESSION"]['ACT_ALGORITHM']:
                    session[session["DIS_SESSION"]['ACT_ALGORITHM']] = {'session_id': None,
                                                            'session_name': None,
                                                            'status': None,
                                                            'Report_enable': False
                                                            }
                    session['prev_algo_name'] = session["DIS_SESSION"]['ACT_ALGORITHM']
                if session["DIS_SESSION"]['ALGORITHM_CATEGORY'] in self.ML_MODULE_LIST:
                    session['ml_run_mode'] = st.selectbox(
                        label= f"#### {item_caption['msg_sel_neworreview'][session['LANG']]}",
                        options=self.MODEL_RUN_OPT,
                        index=None, key='cm_rm')
                    dilogger.debug({sys._getframe().f_code.co_name:f"{session['ml_run_mode']} SELECTED"})
                    if session['ml_run_mode'] == 'Create new model':
                        session['btn_fit_disable'] = False
                    elif session['ml_run_mode'] == 'Review models':
                        session['btn_fit_disable'] = True

    def run_manager(self):
        self.init_model_history()

        with st.sidebar:
            st.markdown("<h1 style='text-align: center; color: black;'>DataIgnite MachineLearning Platform</h1>", unsafe_allow_html=True)
            st.image("./icon/DataIgniteLab_icon.jpg", width=300)
            self.configure_menu()
            if session["DIS_SESSION"]['ALGORITHM_CATEGORY'] == 'Statistics':
                if session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Descriptive':
                    MQTT_Q_Name = 'Statistics_call'
                    if 'BTN_RPT_DISABLE' not in session:
                        session['BTN_RPT_DISABLE'] = True
                    train_df = self.validation_train_df(session["DIS_SESSION"]['TRAIN_DS'])
                    target_columns = self.check_validation_statistic(train_df)
                    model_name = st.text_input(f"#### {item_caption['name_model'][session['LANG']]}",
                                               value=Path(session["DIS_SESSION"]['TRAIN_DS']).stem)
                    if st.button(f"{item_caption['do_analysis'][session['LANG']]}"):
                        self.pub_statistic_arg(MQTT_Q_Name, session["DIS_SESSION"]['TRAIN_DS'], model_name, target_columns)
                        session['BTN_RPT_DISABLE'] = False
                    col_report_result, col_drop_report = st.columns((1, 1))
                    with col_report_result:
                        if st.button(f"{item_caption['call_report'][session['LANG']]}",
                                     disabled=session['BTN_RPT_DISABLE']):
                            sid, status, msg = self.get_report_enable()
                            if sid is None:
                                if status == 'no service':
                                    st.error('프로세스가 작동 중이 아닙니다.')
                            elif status == 'finish':
                                st.write(f"Session {sid}이 처리 완료 되었습니다.")
                            elif status == 'error':
                                st.error(f"Session {sid}의 처리 중 오류가 발생하였습니다.")
                                st.error(f"{msg}")
                    with col_drop_report:
                        pass

            # menu level에 따라 동일 코드를 공통화 하기 위해 알고리즘을 나눈다.
            elif session["DIS_SESSION"]['ALGORITHM_CATEGORY'] in self.ML_MODULE_LIST:
                if session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Randomforest Classifier':
                    st.image("./icon/randomforest_image.png")
                    MQTT_Q_Name = 'RandomforestClassifier_Call'
                    if session['ml_run_mode'] == 'Create new model':
                        st.markdown("---")
                        st.write(f"## {item_caption['parameter'][session['LANG']]}")
                        if 'BTN_RPT_DISABLE' not in session:
                            session['BTN_RPT_DISABLE'] = True
                        train_df = self.validation_train_df(session["DIS_SESSION"]['TRAIN_DS'])
                        self.input_field, self.class_field = self.get_classification_feature_class(train_df)
                        model_name = st.text_input('Name of Model', value=Path(session["DIS_SESSION"]['TRAIN_DS']).stem)
                        fit_parameter, arg_status = self.check_validation_rfc(train_df, model_name)

                elif session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Randomforest Regression':
                    st.image("./icon/randomforest_image.png")
                    MQTT_Q_Name = 'RandomforestRegression_Call'
                    if session['ml_run_mode'] == 'Create new model':
                        st.markdown("---")
                        st.write(f"## {item_caption['parameter'][session['LANG']]}")
                        if 'BTN_RPT_DISABLE' not in session:
                            session['BTN_RPT_DISABLE'] = True
                        train_df = self.validation_train_df(session["DIS_SESSION"]['TRAIN_DS'])
                        self.input_field, self.class_field = self.get_regression_feature_class(train_df)

                        model_name = st.text_input('Name of Model', value=Path(session["DIS_SESSION"]['TRAIN_DS']).stem)
                        fit_parameter, arg_status = self.check_validation_rfr(train_df, model_name)

                if st.button(f"Fit", disabled=session['btn_fit_disable']):
                    if arg_status['class_status'] == 'success' and arg_status['feature_status'] == 'success':
                        self.pub_ml_train_arg(fit_parameter, session["DIS_SESSION"]['TRAIN_DS'], MQTT_Q_Name, model_name)
                        session['BTN_RPT_DISABLE'] = False
                    else:
                        self.disp_error_rpt(arg_status)
                btn_rpt_perform = st.button('Report Model Performance',
                                            disabled=session['BTN_RPT_DISABLE'])  # 전체 공통
                if btn_rpt_perform:
                    sid, status, msg = self.get_report_enable()
                    if sid is None:
                        if status == 'no service':
                            st.error('프로세스가 작동 중이 아닙니다.')
                    elif status == 'success':
                        st.write(f"Session {sid}이 처리 완료 되었습니다.")
                    elif status == 'error':
                        st.error(f"Session {sid}의 처리 중 오류가 발생하였습니다.")
                        st.error(f"{msg}")

            st.markdown("---")
            st.markdown('Setting')

        # Review and Report
        if session['ml_run_mode'] == 'Review Model':  # ML용
            self.view_history()
        else:
            if session["DIS_SESSION"]['TRAIN_DS'] is not None:
                # Review
                st.write(f"#### {item_caption['data_preview'][session['LANG']]}")
                select_df = load_dataset_minio(DS_BUCKET_NAME, session["DIS_SESSION"]['TRAIN_DS'], limit_rows=500)
                st.write("데이터 미리 보기에서는 데이터를 수정할 수 없습니다.")
                st.write(session["DIS_SESSION"]['TRAIN_DS'])
                st.dataframe(select_df)

                input_dname, input_ddesc = '', ''
                ds_store_colnm = Path(session["DIS_SESSION"]['TRAIN_DS']).stem
                ds_store_collec = self.mongo_db.get_collection(f"ds_store_{ds_store_colnm}")
                if ds_store_collec is not None:
                    desc_docs = ds_store_collec.find({})
                    for desc_doc in desc_docs:
                        input_dname = desc_doc['ds_name']
                        input_ddesc = desc_doc['describe']
                else:
                    input_dname = session["DIS_SESSION"]['TRAIN_DS']
                    input_ddesc = ''
                st.write("데이터 이름")
                st.code(input_dname, language="markdown")
                st.write("데이터 설명")
                st.code(input_ddesc, language="markdown")

                st.write('---')
                st.write(f"#### {item_caption['data_preview_preprocess'][session['LANG']]}")
                session['preprocess_df'] = select_df
                st.write("전처리 데이터 미리 보기에서는 데이터를 셀에서도 수정할 수 있습니다.")
                prep_opt = {'특성(필드)추출': 0, '데이터 타입 변경': 1, '값 변경': 2, '컬럼 헤더 추가': 3}
                preprocess_opt = st.selectbox(label=f"{item_caption['convert_fun'][session['LANG']]}",
                                              options=prep_opt.keys(), index=0, key='sel_cvt_method')
                menu_no = prep_opt[preprocess_opt]
                if menu_no == 0:
                    self.extract_column(session['preprocess_df'])
                elif menu_no == 1:
                    self.convert_datatype(session['preprocess_df'])
                elif menu_no == 2:
                    self.replace_column_value(session['preprocess_df'])
                elif menu_no == 3:
                    self.insert_header(session['preprocess_df'])

                rows = session['preprocess_df'].shape[0]
                row_num_list = [str(ri) for ri in range(0, rows)]
                start_row, end_row = st.select_slider(
                    f"{item_caption['row_range'][session['LANG']]}",
                    options=row_num_list,
                    value=('0', str(rows - 1)))
                df_sel_range = session['preprocess_df'].iloc[int(start_row):int(end_row) + 1]

                md_df = st.data_editor(df_sel_range)
                self_df_dtype = pd.DataFrame(session['preprocess_df'].dtypes).transpose()
                self_df_dtype.replace(['object'], '문자형', inplace=True)
                self_df_dtype.replace(['int64'], '정수형', inplace=True)
                self_df_dtype.replace(['float64'], '실수형', inplace=True)
                st.dataframe(self_df_dtype)
                st.dataframe(md_df.describe())
                preprocess_ds_name = Path(session["DIS_SESSION"]['TRAIN_DS']).stem
                preprocess_ds_name = st_keyup(label=f"{item_caption['rename_ds'][session['LANG']]}",
                                              value=f'{preprocess_ds_name}_Prep', on_change=self.update_rn_ds,
                                              key='update_rn_ds')
                self.preprocessed_data = preprocess_ds_name + '.csv'
                st.button(f"{item_caption['upload_prep_ds'][session['LANG']]}", on_click=self.update_ds_list,
                          args=[md_df])

                if session['update_ds_rtn'] is not None:
                    if session['update_ds_rtn'] == True:
                        st.write('Upload success')
                    else:
                        st.error('Upload error')
                if self.class_field is not None:
                    if st.checkbox(f"### {item_caption['show_pca'][session['LANG']]}"):
                        if (len(self.input_field) > 0) and (len(self.class_field)) > 0:
                            prep_col_list = session['preprocess_df'].columns.tolist()
                            if set(self.input_field).issubset(prep_col_list)==True:
                                if set(self.class_field).issubset(prep_col_list)==False:
                                    merged_select_df = self.merget_df_bycolumn(session['preprocess_df'], self.input_field,
                                                                               self.class_field)
                                    nominal_cols = []
                                    for col in self.input_field:
                                        if str(merged_select_df[col].dtypes) == 'object':
                                            nominal_cols.append(col)
                                    if len(nominal_cols) > 0:
                                        st.write(f"{item_caption['nominal_value_er_msg'][session['LANG']]}")
                                    else:
                                        self.draw_heatmap(merged_select_df)
                                        pca_input_df = self.normalization(merged_select_df)
                                        self.show_pca(pca_input_df, self.class_field, nc=len(self.input_field))
                                else:
                                    st.error('주성분 분석 데이터에 클래스 컬럼이 누락되었습니다.')
                                    st.write(f'전처리 대상 컬럼 {prep_col_list}')
                                    st.write(f'클래스 컬럼 {self.class_field}')
                            else:
                                st.error('주성분 분석 데이터에 특성 컬럼이 누락되었습니다.')
                                st.write(f'전처리 대상 컬럼 {prep_col_list}')
                                st.write(f'특성 컬럼 {self.input_field}')

                if session["DIS_SESSION"]['ACT_ALGORITHM'] is not None:
                    aa = session["DIS_SESSION"]['ACT_ALGORITHM']
                    ml_session = session[aa]

                    if session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Descriptive':
                        if ml_session['Report_enable'] is True:
                            sid = session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_id']
                            if sid is not None:
                                do_statistics_report(dilogger, sid, flexible=True)

                                if st.button(f"{item_caption['remove_model'][session['LANG']]}"):
                                    session['BTN_RPT_DISABLE'] = True
                                    self.drop_model()

                    elif session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Randomforest Classifier':
                        if ml_session['Report_enable'] is True:
                            sid = session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_id']
                            do_rfc_report(dilogger, sid, flexible=False)

                            dtn_redis_clear = st.button(f"{item_caption['remove_model'][session['LANG']]}")
                            if dtn_redis_clear == True:
                                sid = self.drop_model()
                                session['BTN_RPT_DISABLE'] = True
                            if sid is not None:
                                st.write(
                                    f"### {session["DIS_SESSION"]['ACT_ALGORITHM']} {item_caption['test_predict'][session['LANG']]}")
                                st.write(f"{item_caption['session_id'][session['LANG']]}: {sid}")
                                if st.button(f"{item_caption['test_model'][session['LANG']]}"):
                                    st.write(f"{item_caption['session_name'][session['LANG']]}",
                                             session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_name'])
                                    self.test_rfc_model(sid)
                    elif session["DIS_SESSION"]['ACT_ALGORITHM'] == 'Randomforest Regression':
                        if ml_session['Report_enable'] is True:
                            sid = session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_id']
                            do_rfr_report(dilogger, sid, flexible=False)

                            dtn_redis_clear = st.button(f"{item_caption['remove_model'][session['LANG']]}")
                            if dtn_redis_clear == True:
                                sid = self.drop_model()
                                session['BTN_RPT_DISABLE'] = True
                            if sid is not None:
                                st.write(
                                    f"### {session["DIS_SESSION"]['ACT_ALGORITHM']} {item_caption['test_predict'][session['LANG']]}")
                                st.write(f"{item_caption['session_id'][session['LANG']]}: {sid}")
                                if st.button(f"{item_caption['test_model'][session['LANG']]}"):
                                    st.write(f"{item_caption['session_name'][session['LANG']]}",
                                             session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_name'])
                                    self.test_rfr_model(sid)


    def get_file_name(self):
        pass
    def drop_model(self):
        sid = session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_id']

        if session['ACT_ALGORITHM'] == 'Randomforest Classifier':
            key = f'rfc_status_{sid}'
            redis_mgr.delete(key)
            key = f'rfc_{sid}'
            redis_mgr.delete(key)
        elif session['ACT_ALGORITHM'] == 'Statistics':
            key = f"statistics_status_{session['Statistics']['session_id']}"
            redis_mgr.delete(key)
            key = f"statistics_{session['Statistics']['session_id']}"
            redis_mgr.delete(key)
        if redis_mgr.get(key) is None: st.write(f'Session {sid} Remove')
        session[session["DIS_SESSION"]['ACT_ALGORITHM']]['session_id'] = None
        return None

    def normalization(self, data_df):
        scaler = MinMaxScaler()
        data_scale = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns, index=data_df.index)
        return data_scale

    def show_pca(self, data_df, target, nc=None):
        if nc is None:
            nc = data_df.shape[1]

        scikit_pca = PCA(n_components=nc)
        component_df = scikit_pca.fit_transform(data_df)

        pca_column = [f'pca_{c}' for c in range(nc)]
        df_pca = pd.DataFrame(data=component_df, columns=pca_column)
        df_pca['target'] = data_df[target]
        st.write(f'PCA {nc}')
        st.dataframe(df_pca)
        pca_report = pd.DataFrame({'Explained Variance': scikit_pca.explained_variance_,
                                   'Variance Ratio': scikit_pca.explained_variance_ratio_},
                                  index=np.array([f'pca{c}' for c in range(nc)]))
        pca_report['Accumulate Variance Ratio'] = pca_report['Variance Ratio'].cumsum()
        st.dataframe(pca_report)

        model = pca(n_components=nc)
        model.fit_transform(data_df)
        width = 12.0
        height = 4.0
        fig, ax = model.biplot(n_feat=nc, legend=False, figsize=(width, height))
        st.write(fig)

    def check_missing_data(self, df):
        nan_df = df[df.isna().any(axis=1)]
        if nan_df.shape[0] > 0:
            st.caption(f"### {item_caption['detected_missing_data'][session['LANG']]}")
            st.dataframe(nan_df)

    def get_report_enable(self):
        ml_status = session[session['DIS_SESSION']['ACT_ALGORITHM']]
        sid = ml_status['session_id']
        model_status = None
        if session['DIS_SESSION']['ACT_ALGORITHM'] == 'Randomforest Classifier':
            model_status = redis_mgr.get(f"rfc_{sid}")

        elif session['DIS_SESSION']['ACT_ALGORITHM'] == 'Randomforest Regression':
            model_status = redis_mgr.get(f"rfr_{sid}")

        elif session['DIS_SESSION']['ACT_ALGORITHM'] == 'Descriptive':
            model_status = redis_mgr.get(f"statistics_{sid}")

        if model_status is None:
            return sid, 'no service', None
        else:
            session[session['DIS_SESSION']['ACT_ALGORITHM']]['Report_enable'] = True
            dilogger.error({"status": "ok", sys._getframe().f_code.co_name: f"model_status:{model_status}"})
            json_model_status = json.loads(model_status)
            status = json_model_status['status']
            if status == 'error':
                msg = json_model_status['msg']
                return sid, status, msg
            elif status == 'finish':
                return sid, status, None
            else:
                if 'msg' in json_model_status:
                    msg = json_model_status['msg']
                else:
                    msg = None
                return sid, status, msg

    def update_session_status(self, active_algo, session_name):
        session[active_algo]['session_id'] = time.time_ns() // 1_000_000
        session[active_algo]['session_name'] = session_name
        session[active_algo]['status'] = 'pcall'
        session[active_algo]['Report_enable'] = False

    def init_model_history(self):
        if 'model_history' not in session:
            algorith_type = {'Statistics': dict(),
                             'Randomforest Classifier': dict()}
            session['model_history'] = algorith_type

    def draw_heatmap(self, corr_df):
        st.write('Correlation Heat Map')
        width = 12.0
        height = 4.0
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(corr_df.corr(), ax=ax)
        st.write(fig)

    def get_regression_feature_class(self, train_df):
        column_list = train_df.columns.tolist()
        default_value = get_ValuesofWidget(session['DIS_SESSION'], column_list, 'REGRESSION_INPUT')
        stored_default = []
        if set(default_value).issubset(set(column_list)) == True:
            stored_default = default_value
        else:
            stored_default = column_list
        st.multiselect(f"##### {item_caption['features_field'][session['LANG']]}",
                       column_list,
                       default=stored_default,
                       key='REGRESSION_INPUT',
                       on_change=self.on_change_regression_input)
        sel_input = upsert_session(session['DIS_SESSION'], 'REGRESSION_INPUT', session.REGRESSION_INPUT)
        sel_input = sel_input['REGRESSION_INPUT']

        self.input_field = [col for col in column_list if col in sel_input]
        output_can_list = list(set(column_list) - set(self.input_field))
        opt_index = get_IndexofWidget(session['DIS_SESSION'], output_can_list, 'REGRESSION_OUTPUT')
        st.selectbox(f"##### {item_caption['class_field'][session['LANG']]}", output_can_list,
                                   index=opt_index,
                                   key='REGRESSION_OUTPUT',
                                   on_change=self.on_change_regression_output)
        out_field = upsert_session(session['DIS_SESSION'], 'REGRESSION_OUTPUT', session.REGRESSION_OUTPUT)
        out_field = out_field['REGRESSION_OUTPUT']
        return self.input_field, out_field

    def get_classification_feature_class(self, train_df):
        column_list = train_df.columns.tolist()
        default_value = get_ValuesofWidget(session['DIS_SESSION'], column_list, 'CLASSIFICATION_INPUT')
        if set(default_value).issubset(set(column_list)) == True:
            stored_default = default_value
        else:
            stored_default = column_list
        st.multiselect(f"##### {item_caption['features_field'][session['LANG']]}",
                       column_list,
                       default=stored_default,
                       key='CLASSIFICATION_INPUT',
                       on_change=self.on_change_classification_input)
        sel_input = upsert_session(session['DIS_SESSION'], 'CLASSIFICATION_INPUT', session.CLASSIFICATION_INPUT)
        sel_input = sel_input['CLASSIFICATION_INPUT']
        self.input_field = [col for col in column_list if col in sel_input]
        output_can_list = list(set(column_list) - set(self.input_field))

        opt_index = get_IndexofWidget(session['DIS_SESSION'], output_can_list, 'CLASSIFICATION_OUTPUT')
        st.selectbox(f"##### {item_caption['class_field'][session['LANG']]}",
                                   output_can_list,
                                   index=opt_index,
                                   key='CLASSIFICATION_OUTPUT',
                                   on_change=self.on_change_classification_output)
        class_field = upsert_session(session['DIS_SESSION'], 'CLASSIFICATION_OUTPUT', session.CLASSIFICATION_OUTPUT)
        class_field = class_field['CLASSIFICATION_OUTPUT']

        return self.input_field, class_field

    def get_rfc_fit_parameter(self, model_name):
        numbers_of_tree = st.number_input(f"#### {item_caption['rfc_numtrees'][session['LANG']]}", min_value=1,
                                          max_value=100, value=5,
                                          placeholder="Type a number...")
        max_dept = st.number_input(f"#### {item_caption['rfc_maxdepth'][session['LANG']]}", value=3,
                                   placeholder='Type a number...')
        max_features = st.number_input(f"#### {item_caption['rfc_maxfeatures'][session['LANG']]}", value=3,
                                       placeholder='Type a number...')
        max_leaf_nodes = st.number_input(f"#### {item_caption['rfc_maxleaf_nodes'][session['LANG']]}", value=3,
                                         placeholder='Type a number...')
        min_samples_leaf = st.number_input(f"#### {item_caption['rfc_min_samples_leaf'][session['LANG']]}", value=3,
                                           placeholder='Type a number...')
        min_samples_split = st.number_input(f"#### {item_caption['rfc_min_samples_split'][session['LANG']]}",
                                            value=3,
                                            placeholder="Type a number...")
        min_weight_fraction = st.number_input(f"#### {item_caption['rfc_min_weight_frac'][session['LANG']]}",
                                              value=3,
                                              placeholder="Type a number...")
        n_estimators = st.number_input(f"#### {item_caption['rfc_num_estimator'][session['LANG']]}", value=10,
                                       placeholder='Type a number...')
        test_size = st.number_input(f"#### {item_caption['rfc_test_size'][session['LANG']]}", value=10,
                                    placeholder='Type a number...')
        criterion = st.selectbox(f"#### {item_caption['rfc_Criterion'][session['LANG']]}",
                                 options=['gini', 'entropy', 'log_loss'], index=0)
        fit_parameter = {
            'x_name_list': self.input_field,
            'target_name': self.class_field,
            'model_name': model_name,
            'numbers_of_tree': numbers_of_tree,
            'max_dept': max_dept,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'criterion': criterion,
            'min_weight_fraction': min_weight_fraction,
            'n_estimators': n_estimators,
            'test_size': test_size
        }
        return fit_parameter

    def get_rfr_fit_parameter(self, model_name):
        numbers_of_tree = st.number_input(f"#### {item_caption['rfr_numtrees'][session['LANG']]}", min_value=1,
                                          max_value=100, value=5,
                                          placeholder="Type a number...")
        criterion = st.selectbox(f"#### {item_caption['rfr_Criterion'][session['LANG']]}",
                                 options=["squared_error", "absolute_error", "friedman_mse", "poisson"], index=0)
        max_dept = st.number_input(f"#### {item_caption['rfr_maxdepth'][session['LANG']]}", value=2,
                                   placeholder='Type a number...')
        min_samples_split = st.number_input(f"#### {item_caption['rfc_min_samples_split'][session['LANG']]}",
                                            value=2,
                                            placeholder="Type a number...")
        min_samples_leaf = st.number_input(f"#### {item_caption['rfr_min_samples_leaf'][session['LANG']]}", value=2,
                                           placeholder='Type a number...')
        min_weight_fraction = st.number_input(f"#### {item_caption['rfr_min_weight_frac'][session['LANG']]}",
                                              value=0.0,
                                              placeholder="Type a number...")

        max_features = st.number_input(f"#### {item_caption['rfr_maxfeatures'][session['LANG']]}", value=1.0,
                                       placeholder='Type a number...')
        max_leaf_nodes = st.number_input(f"#### {item_caption['rfc_maxleaf_nodes'][session['LANG']]}", value=2,
                                         placeholder='Type a number...')
        min_impurity_decrease = st.number_input(f"#### {item_caption['rfr_min_impurity_decrease'][session['LANG']]}", value=0.0,
                                         placeholder='Type a number...')

        fit_parameter = {
            'x_name_list': self.input_field,
            'y_name': self.class_field,
            'model_name': model_name,
            'n_estimators': numbers_of_tree,
            'criterion': criterion,
            'max_dept': max_dept,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction': min_weight_fraction,
            'max_features': max_features,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease': min_impurity_decrease
        }
        return fit_parameter

    def get_pika_channel(self):
        connection = pika.BlockingConnection(self.pika_conn_param)
        process_call_channel = connection.channel()
        return process_call_channel

    def check_validation_statistic(self, train_df):
        column_list = train_df.columns.tolist()
        stored_default_list = []
        default_value = get_ValuesofWidget(session['DIS_SESSION'], column_list, 'STATISTICS_DESC_INPUT')
        if set(default_value).issubset(set(column_list)) == True:
            stored_default_list = default_value
        else:
            stored_default_list = column_list
        st.multiselect(f"#### {item_caption['features_field'][session['LANG']]}",
                       column_list,
                       default=stored_default_list,
                       key='STATISTICS_DESC_INPUT',
                       on_change=self.on_change_statistics_descriptive_input)
        sel_input = upsert_session(session['DIS_SESSION'], 'STATISTICS_DESC_INPUT', session.STATISTICS_DESC_INPUT)
        sel_input = sel_input['STATISTICS_DESC_INPUT']
        filter = st.radio(f"{item_caption['sel_opt'][session['LANG']]}", ("선택 커럼 포함", "선택 컬럼 제거"),
                          key='cvs_radio')

        if filter == '선택 커럼 포함':
            filter_in_columns = [col for col in column_list if col in sel_input]
        elif filter == '선택 컬럼 제거':
            filter_in_columns = [col for col in column_list if col not in sel_input]
        if len(filter_in_columns) == 0:
            return None
        else:
            return filter_in_columns

    def check_validation_rfc(self, train_df, model_name):
        arg_status = {'class_status': 'success',
                      'feature_status': 'success'}
        isClassValid = True
        fit_parameter = self.get_rfc_fit_parameter(model_name)
        if self.class_field is None:
            arg_status['class_status'] = {'error': f"클래스 필드를 선택하세요"}
            isClassValid = False
        else:
            class_labels = train_df[self.class_field].unique()
            if len(class_labels) == 1:
                if session['ACT_ALGORITHM'] == 'Randomforest Classifier':
                    arg_status['class_status'] = {
                        'error': f"1 개의 클래스만 존재합니다. 다중 분류 알고리즘을 수행할 수 없습니다."}

        nominal_cols = []
        for col in self.input_field:
            if str(train_df[col].dtypes) == 'object':
                nominal_cols.append(col)
        if len(nominal_cols) > 0:
            arg_status['feature_status'] = {
                'error': f"{nominal_cols}에서 숫자가 아닌 값을 특성 값을 발견하였습니다."}
        if isClassValid == True:
            if len(self.input_field) == 0:
                arg_status['feature_status'] = {
                    'error': f"특성 필드가 비어 있습니다."}
                isClassValid = False
            if len(self.class_field) == 0:
                arg_status['feature_status'] = {
                    'error': f"클래스 필드가 비어 있습니다."}
                isClassValid = False

            if fit_parameter['max_features'] > len(self.input_field):
                arg_status['feature_status'] = {
                    'error': f"트래의 최대 특성 개수는 입력 특성의 개수를 초과할 수 없습니다."}
                isClassValid = False
            if isClassValid == True:
                return fit_parameter, arg_status
            else:
                return fit_parameter, arg_status
        else:
            return fit_parameter, arg_status

    def check_validation_rfr(self, train_df, model_name):
        arg_status = {'class_status': 'success',
                      'feature_status': 'success'}
        isClassValid = True
        fit_parameter = self.get_rfr_fit_parameter(model_name)
        if self.class_field is None:
            arg_status['class_status'] = {'error': f"클래스 필드를 선택하세요"}
            isClassValid = False
        else:
            class_labels = train_df[self.class_field].unique()
            if len(class_labels) == 1:
                if session['ACT_ALGORITHM'] == 'Randomforest Classifier':
                    arg_status['class_status'] = {
                        'error': f"1 개의 클래스만 존재합니다. 다중 분류 알고리즘을 수행할 수 없습니다."}

        nominal_cols = []
        for col in self.input_field:
            if str(train_df[col].dtypes) == 'object':
                nominal_cols.append(col)
        if len(nominal_cols) > 0:
            arg_status['feature_status'] = {
                'error': f"{nominal_cols}에서 숫자가 아닌 값을 특성 값을 발견하였습니다."}
        if isClassValid == True:
            if len(self.input_field) == 0:
                arg_status['feature_status'] = {
                    'error': f"특성 필드가 비어 있습니다."}
                isClassValid = False
            if len(self.class_field) == 0:
                arg_status['feature_status'] = {
                    'error': f"클래스 필드가 비어 있습니다."}
                isClassValid = False

            if fit_parameter['max_features'] > len(self.input_field):
                arg_status['feature_status'] = {
                    'error': f"트래의 최대 특성 개수는 입력 특성의 개수를 초과할 수 없습니다."}
                isClassValid = False
            if isClassValid == True:
                return fit_parameter, arg_status
            else:
                return fit_parameter, arg_status
        else:
            return fit_parameter, arg_status

    def pub_train(self, channel, fit_parameter, train_ds_obj, q_name):
        sid = session[session['DIS_SESSION']['ACT_ALGORITHM']]['session_id']
        process_call_msg = {'session_id': sid,
                            'model_name': session[session['DIS_SESSION']['ACT_ALGORITHM']]['session_name'],
                            'timestamp': str(datetime.now()),
                            'bucket_name': DS_BUCKET_NAME,
                            'input': train_ds_obj,
                            'fit_param': fit_parameter}
        ml_status = session[session['DIS_SESSION']['ACT_ALGORITHM']]
        ml_status['Report_enable'] = False
        channel.basic_publish(exchange='', routing_key=q_name,
                              body=json.dumps(process_call_msg))
        return sid

    def disp_error_rpt(self, arg_status):
        if arg_status['class_status'] != 'success':
            er_msg = arg_status['class_status']['error']
            st.error(er_msg, icon='🚨')
            session['BTN_RPT_DISABLE'] = True
        if arg_status['feature_status'] != 'success':
            er_msg = arg_status['feature_status']['error']
            st.error(er_msg, icon='🚨')
            session['BTN_RPT_DISABLE'] = True

    def validation_train_df(self, ds_name):
        train_df = load_dataset_minio(DS_BUCKET_NAME, ds_name)
        self.check_missing_data(train_df)
        return train_df

    def pub_ml_train_arg(self, fit_parameter, train_ds_name, q_name, model_name):
        call_channel = self.get_pika_channel()
        call_channel.queue_declare(queue=q_name)
        self.update_session_status(active_algo=session['DIS_SESSION']['ACT_ALGORITHM'], session_name=model_name)
        sid = self.pub_train(call_channel, fit_parameter, train_ds_name, q_name)
        st.write(f"The {session['DIS_SESSION']['ACT_ALGORITHM']} process was called.")
        st.write(f"Session ID : {sid}")
        return sid

    def pub_statistic_arg(self, q_name, train_ds_name, model_name, target_columns):
        connection = pika.BlockingConnection(self.pika_conn_param)
        call_channel = connection.channel()
        call_channel.queue_declare(queue=q_name)
        self.update_session_status(active_algo=session['DIS_SESSION']['ACT_ALGORITHM'], session_name=model_name)
        pprint.pprint(session['DIS_SESSION']['ACT_ALGORITHM'])
        sid = self.pub_train(call_channel, {'x_name_list': target_columns}, train_ds_name, q_name)
        st.write(f"The {session['DIS_SESSION']['ACT_ALGORITHM']} process was called.")
        st.write(f'Session ID : {sid}')

    def test_rfc_model(self, sid):
        model_obj, test_xt, test_yt, test_yp = get_rfc_model(sid)
        test_xt = test_xt.reset_index(drop=True)
        test_yt = test_yt.reset_index(drop=True)
        predict_np = model_obj.predict(test_xt)
        yt_df = pd.DataFrame(test_yt)
        predict_df = pd.DataFrame(predict_np, columns=['Predict'])
        result_df = pd.concat([test_xt, yt_df, predict_df], axis=1)
        st.dataframe(result_df)

    def test_rfr_model(self, sid):
        model_obj, test_xt, test_yt, test_yp = get_rfr_model(sid)
        test_xt = test_xt.reset_index(drop=True)
        test_yt = test_yt.reset_index(drop=True)
        predict_np = model_obj.predict(test_xt)
        yt_df = pd.DataFrame(test_yt)
        predict_df = pd.DataFrame(predict_np, columns=['Predict'])
        result_df = pd.concat([test_xt, yt_df, predict_df], axis=1)
        st.dataframe(result_df)

    def update_rn_ds(self):
        session['update_rename_ds'] = session.update_rn_ds

    def update_ds_list(self, md_df):
        md_df_csv = md_df.reset_index(drop=True).to_csv(index=False).encode('utf-8')
        if session['update_rename_ds'] is not None:
            self.preprocessed_data = session['update_rename_ds']
        minio_client.put_object(
            DS_BUCKET_NAME,
            self.preprocessed_data,
            data=io.BytesIO(md_df_csv),
            length=len(md_df_csv),
            content_type='application/csv'
        )
        session['list_dataset'] = get_files_minio(DS_BUCKET_NAME)
        if self.preprocessed_data in session['list_dataset']:
            session['update_ds_rtn'] = True
        else:
            session['update_ds_rtn'] = False

    def merget_df_bycolumn(self, select_df, input_field, class_field):
        in_df = select_df[self.input_field]
        out_df = select_df[self.class_field]
        merged_select_df = pd.concat([in_df, out_df], axis=1)
        st.dataframe(merged_select_df)
        return merged_select_df

    def convert_datatype(self, train_df):
        if session['preprocess_df'] is None:
            cvt_df = train_df.copy(deep=True)
        else:
            cvt_df = session['preprocess_df']

        # modify proprocess_df
        candidate_col = cvt_df.columns.tolist()
        null_row_cnt = cvt_df.isnull().any(axis=1).sum()
        if null_row_cnt > 0:
            st.info(f'{null_row_cnt} rows will Null Drop')
        cvt_df = cvt_df.dropna(axis=0)
        target_cols = st.multiselect(label=f"{item_caption['cols_type_conver'][session['LANG']]}",
                                     options=candidate_col, default=None)
        cvt_opt = ['To string', 'To integer', 'To float', 'Label Encoding']
        if len(target_cols) > 0:
            conversion_type = st.selectbox(label=f"{item_caption['convert_fun'][session['LANG']]}", options=cvt_opt,
                                           index=None)
            print(target_cols, conversion_type)
            if conversion_type == 'To string':
                try:
                    for target_col in target_cols:
                        cvt_df[target_col] = cvt_df[target_col].astype('string')
                except Exception as e:
                    st.error(str(e))

            elif conversion_type == 'To integer':
                try:
                    for target_col in target_cols:
                        cvt_df[target_col] = cvt_df[target_col].apply(pd.to_numeric)
                        cvt_df[target_col] = cvt_df[target_col].astype('int')
                except Exception as e:
                    st.error(str(e))
            elif conversion_type == 'To float':
                try:
                    for target_col in target_cols:
                        cvt_df[target_col] = cvt_df[target_col].apply(pd.to_numeric)
                        cvt_df[target_col] = cvt_df[target_col].astype('float')
                except Exception as e:
                    st.error(str(e))
            elif conversion_type == 'Label Encoding':
                lencoder = LabelEncoder()
                try:
                    for target_col in target_cols:
                        lencoder = lencoder.fit(cvt_df[target_col])
                        cvt_df[target_col] = lencoder.transform(cvt_df[target_col])
                except Exception as e:
                    st.error(str(e))
            if target_cols != None and conversion_type != None:
                session['preprocess_df'] = cvt_df

    def insert_header(self, train_df):
        if session['preprocess_df'] is None:
            cvt_df = train_df.copy(deep=True)
        else:
            cvt_df = session['preprocess_df']
        st.text_input('기본 컬럼 이름', 'Col', key='ins_header_col')
        if st.button('컬럼 이름 적용'):
            if session.ins_header_col is None: return
            if len(session.ins_header_col) == 0: return
            default_name = session.ins_header_col
            default_col_name = []
            for ci in range(cvt_df.shape[1]):
                default_col_name.append(f'{default_name}_{ci}')
            cvt_df.columns = default_col_name
            session['preprocess_df'] = cvt_df

    def extract_column(self, train_df):
        if session['preprocess_df'] is None:
            cvt_df = train_df.copy(deep=True)
        else:
            cvt_df = session['preprocess_df']

        # modify proprocess_df
        candidate_col = cvt_df.columns.tolist()
        null_row_cnt = cvt_df.isnull().any(axis=1).sum()
        if null_row_cnt > 0:
            st.info(f'{null_row_cnt} rows will Null Drop')
        cvt_df = cvt_df.dropna(axis=0)
        target_cols = st.multiselect(label=f"{item_caption['drop_col'][session['LANG']]}", options=candidate_col,
                                     default=None)
        if len(target_cols) > 0:
            session['preprocess_df'] = cvt_df[target_cols]

    def replace_column_value(self, train_df):
        if session['preprocess_df'] is None:
            cvt_df = train_df.copy(deep=True)
        else:
            cvt_df = session['preprocess_df']

        # modify proprocess_df
        candidate_col = cvt_df.columns.tolist()
        null_row_cnt = cvt_df.isnull().any(axis=1).sum()
        if null_row_cnt > 0:
            st.info(f'{null_row_cnt} rows will Null Drop')
        cvt_df = cvt_df.dropna(axis=0)
        target_cols = st.multiselect(label=f"{item_caption['replace_col'][session['LANG']]}", options=candidate_col,
                                     default=None)
        if len(target_cols) > 0:
            target_rp_value = []
            for target_col in target_cols:
                target_rp_value.extend(cvt_df[target_col].unique().tolist())
            org_value = st.multiselect(label=f"{item_caption['origin_value'][session['LANG']]}",
                                       options=target_rp_value, default=None)
            rp_value = st.text_input(label=f"{item_caption['rep_value'][session['LANG']]}", value='')
            if st.button('Apply'):
                if len(rp_value) > 0:
                    for target_col in target_cols:
                        print(target_col, org_value, rp_value)
                        cvt_df[target_col] = cvt_df[target_col].replace(org_value, rp_value)
                    if target_cols != None and rp_value != None:
                        session['preprocess_df'] = cvt_df

    def preprocess_isorigin(self, select_df):
        if session['preprocess_df'] is None:
            return False
        else:
            return select_df.equals(session['preprocess_df'])


if __name__ == "__main__":
    st.set_page_config(
        page_title="DataIgnite Machine Learning Platform", page_icon=":chart_with_upwards_trend:", layout="centered"
    )
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown("""<style>
                                div.stButton button {
                                width: 400px;
                                }
                                </style>""", unsafe_allow_html=True)
    wm = WebMain()
    wm.run_manager()
