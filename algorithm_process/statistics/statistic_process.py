import os
import pickle
import time
from datetime import datetime
from io import StringIO
from minio import Minio
import pandas as pd
import pika
import json
import redis
import sys
from dilogger import DIlogger

elk_address = os.environ.get('elk_address')

if elk_address is not None:
    address_list = [elk_address]
else:
    address_list = None
index = 'dataignite-statistics-log'

dilogger = DIlogger(address_list = address_list, index = index, level="DEBUG")

minio_host = os.environ.get('minio_host')
dilogger.info({"status":"ok", "minio host": minio_host})
if minio_host == None:
    client = Minio("localhost:9000", "dataignite", "dataignite", secure=False)
else:
    client = Minio(minio_host, "dataignite", "dataignite", secure=False)


redis_host = os.environ.get('redis_host')
dilogger.info({"status":"ok", "redis_host host": redis_host})
if redis_host == None:
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
else:
    r = redis.StrictRedis(host=redis_host, port=6379, db=0)

class Statistic():
    def __init__(self):
        self.report = []
        self.credentials = pika.PlainCredentials('guest', 'guest')
        mqbroker = os.environ.get('mqbroker')
        if mqbroker == None:
            self.param = pika.ConnectionParameters('localhost', 5672, '/', self.credentials)
        else:
            self.param = pika.ConnectionParameters(mqbroker, 5672, '/', self.credentials)

        self.sub_connection = pika.BlockingConnection(self.param)
        self.sub_channel = self.sub_connection.channel()
        q_name = 'Statistics_call'
        mq = self.sub_channel.queue_declare(queue=q_name).method.queue
        self.sub_channel.basic_consume(mq, on_message_callback=self.wait_mq_signal, auto_ack=True)
        self.sub_channel.start_consuming()


    def wait_mq_signal(self,ch, method, properties, body):
        try:
            body = body.decode('utf-8')
            msg_json = json.loads(body)
            dilogger.info({"status": "ok", "statistic-subscribe": msg_json})
            fit_param = msg_json['fit_param']
            if fit_param is not None:
                dilogger.info({"status": "ok", sys._getframe().f_code.co_name: "fit start"})
                fit_parameter = {
                    'session_id': msg_json['session_id'],
                    'timestamp': msg_json['timestamp'],
                    'bucket_name': msg_json['bucket_name'],
                    'model_name': msg_json['model_name'],
                    'input_path': msg_json['input'],
                    'x_name_list': fit_param['x_name_list'],
                }
                dilogger.info({"status": "ok", sys._getframe().f_code.co_name: f"fit argument:{fit_parameter}"})
                self.fit(**fit_parameter)
                session_id = fit_parameter['session_id']
                report_json = {'status': 'finish',
                            'session_id': str(msg_json['session_id']),
                            'timestamp': msg_json['timestamp'],
                            'report': self.report,
                            'name': msg_json['model_name'],
                            'algorithm': 'Statistics'}
                value_dump_model = pickle.dumps(report_json)
                r.set(f'statistics_report_{session_id}', value_dump_model)
                dilogger.info({"status": "ok", sys._getframe().f_code.co_name: f"finish"})

                # session_key = f'statistics_status_{session_id}'
                # report_status = {'status': 'success',
                #                  'session_id': str(session_id),
                #                  'timestamp': msg_json['timestamp'],
                #                  'name': msg_json['model_name'],
                #                  'algorithm': 'Statistics'}
                # r.set(session_key, json.dumps(report_status, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            session_id = fit_parameter['session_id']
            key_status = f'statistics_{session_id}'
            msg_json = {'status': 'error',
                        'session_id': str(msg_json['session_id']),
                        'timestamp': msg_json['timestamp'],
                        'msg': str(e),
                        'name': msg_json['model_name'],
                        'algorithm': 'Statistics'}
            r.set(key_status, json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
            dilogger.error({"status": "error", sys._getframe().f_code.co_name: f"fit error:{msg_json}"})

        self.report.clear()

    def put_report_msg(self, rpt_describe, rpt_type, rpt_feature, rpt_data):
        if rpt_type == 'df':
            r_data_json = rpt_data.to_json(index=True, force_ascii=False)
            r_data_json = json.loads(r_data_json)
            self.report.append({'describe':rpt_describe,
                                'showtype':('df',r_data_json),
                                'feature':None})
        elif rpt_type == 'hist':
            r_data_json = rpt_data.to_json(index=True)
            self.report.append({'describe':rpt_describe,
                                'showtype':('hist',r_data_json),
                                'feature':rpt_feature})
        elif rpt_type == 'pie':
            r_data_json = rpt_data.to_json(index=True)
            r_data_json = json.loads(r_data_json)
            self.report.append({'describe':rpt_describe,
                                'showtype':('pie',r_data_json),
                                'feature':rpt_feature})
        elif rpt_type == 'bar':
            r_data_json = rpt_data.to_json(index=True)
            r_data_json = json.loads(r_data_json)
            self.report.append({'describe':rpt_describe,
                                'showtype':('bar',r_data_json),
                                'feature':rpt_feature})

    def fit(self, **kargs):
        session_id = kargs['session_id']
        timestamp = kargs['timestamp']
        bucket_name = kargs['bucket_name']
        input_path = kargs['input_path']
        x_name_list = kargs['x_name_list']
        model_name = kargs['model_name']

        object_contents = client.get_object(bucket_name, input_path)
        object_data = object_contents.data
        _, file_type = os.path.splitext(input_path)
        try:
            if 'xls' in file_type:
                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'xls data load start',
                            'name': model_name,
                            'filename': input_path,
                            'algorithm': 'Statistics'}
                r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"xls data load start:{input_path}"})
                input_df = pd.read_excel(object_data)


                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'xls data load success',
                            'name': model_name,
                            'filename': input_path,
                            'algorithm': 'Statistics'}
                r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"xls data load success:{input_path}"})

            elif 'csv':
                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'csv data load start',
                            'name': model_name,
                            'algorithm': 'Statistics'}
                r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"csv data load start:{input_path}"})

                object_str = str(object_data, 'utf-8')
                data = StringIO(object_str)
                input_df = pd.read_csv(data)

                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'csv data load success',
                            'name': model_name,
                            'algorithm': 'Statistics'}
                r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"csv data load success:{input_path}"})

            else:
                msg_json = {'status': 'error', 'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'unknown data type',
                            'name': model_name,
                            'algorithm': 'Statistics'}
                r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"unknown data type:{input_path}"})
                return 0
        except Exception as e:
            msg_json = {'status': 'error',
                        'session_id': str(session_id),
                        'timestamp': timestamp,
                        'msg': str(e),
                        'name': model_name,
                        'algorithm': 'Statistics'}
            r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
            dilogger.error({"status": "error", sys._getframe().f_code.co_name: f"fit data format error:{msg_json}"})
            return 0

        x_integer_list = []
        x_float_list = []
        x_norm_list = []

        for x_name in x_name_list:
            if 'int' in str(input_df.dtypes[x_name]):
                x_integer_list.append(x_name)
            elif 'float' in str(input_df.dtypes[x_name]):
                x_float_list.append(x_name)
            elif 'object' in str(input_df.dtypes[x_name]):
                x_norm_list.append(x_name)
        numeric_list = x_integer_list + x_float_list
        input_statistic = pd.DataFrame(input_df.describe()[numeric_list])
        metrics = ['count', 'max', 'min', 'mean']

        allval = {}
        for metric in metrics:
            for x_name in input_statistic:
                if not x_name in allval:
                    allval[x_name] = {"Field": x_name}
                metric_label = metric
                if metric == 'count':
                    metric_label = 'Total count'
                elif metric == 'max':
                    metric_label = 'Max'
                elif metric == 'min':
                    metric_label = 'Min'
                elif metric == 'mean':
                    metric_label = 'Aveage'
                if metric == 'count':
                    allval[x_name][metric_label] = '{:,.0f}'.format(input_statistic.at[metric, x_name])
                else:
                    allval[x_name][metric_label] = '{:,.3f}'.format(input_statistic.at[metric, x_name])

        input_sem = pd.DataFrame(input_df[input_statistic].sem())
        var_df = pd.DataFrame(input_df[input_statistic].var())
        skew_df = pd.DataFrame(input_df[input_statistic].skew())
        kurtosis_df = pd.DataFrame(input_df[input_statistic].kurtosis())

        for x_name in numeric_list:
            input_statistic_minmax = input_statistic.at['max', x_name] - input_statistic.at['min', x_name]
            allval[x_name]['Range'] = '{:,.3f}'.format(input_statistic_minmax)
            allval[x_name]['Standard Error'] = '{:,.3f}'.format(input_sem.at[x_name, 0])
            allval[x_name]['Variance'] = '{:,.3f}'.format(var_df.at[x_name, 0])
            allval[x_name]['Skewness'] = '{:,.3f}'.format(skew_df.at[x_name, 0])
            allval[x_name]['Kurtosis'] = '{:,.3f}'.format(kurtosis_df.at[x_name, 0])

        tbl = []
        if len(numeric_list) > 0:
            for x_name in numeric_list:
                tbl.append(allval[x_name])
            self.put_report_msg(rpt_describe='수치 데이터 통계', rpt_type='df', rpt_feature=None, rpt_data=pd.DataFrame(tbl))
            dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: "수치 데이터 통계 완료"})

        if len(numeric_list) > 0:
            for x_name in numeric_list:
                self.put_report_msg(rpt_describe='연속 변수 히스토그램',rpt_type='hist',rpt_feature=x_name,  rpt_data=input_df[x_name])
            dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: "연속 변수 히스토그램 완료"})

        if len(x_norm_list) > 0:
            self.put_report_msg(rpt_describe='문자형 데이터 통계',rpt_type='df', rpt_feature=None, rpt_data=input_df[x_norm_list].describe())
            dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: "문자형 데이터 통계 완료"})
            for norm_col in x_norm_list:
                pie_df = input_df[norm_col].head(10).value_counts()
                self.put_report_msg(rpt_describe='파이 차트(문자형 데이터)-Top 10', rpt_type='pie',rpt_feature=norm_col,rpt_data=pie_df)

            for norm_col in x_norm_list:
                bar_df = input_df[norm_col].head(10).value_counts()
                self.put_report_msg(rpt_describe='바 차트(문자형 데이터)-Top 10', rpt_type='bar',rpt_feature=norm_col,rpt_data=bar_df)

        msg_json = {'status': 'finish',
                    'session_id': str(session_id),
                    'timestamp': timestamp,
                    'name': model_name,
                    'algorithm': 'Statistics'}
        r.set(f'statistics_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
        dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"csv data load success:{input_path}"})
if __name__ == "__main__":
    statistic = Statistic()