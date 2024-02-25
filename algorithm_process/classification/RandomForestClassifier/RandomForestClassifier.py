# -*- coding: utf-8 -*-
import pickle
import sys
import pika
import json
import pandas as pd
import os
from io import StringIO
import numpy as np
import redis
from minio import Minio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import logging

from dilogger import DIlogger

elk_address = os.environ.get('elk_address')

if elk_address is not None:
    address_list = [elk_address]
else:
    address_list = None

index = 'dataignite-rfc-log'
dilogger = DIlogger(address_list = address_list, index = index, level="DEBUG")


redis_host = os.environ.get('redis_host')
if redis_host == None:
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    dilogger.info({"Redis Address": f"rabbitmq host:localhost:6379"})
else:
    r = redis.StrictRedis(host=redis_host, port=6379, db=0)
    dilogger.info({"Redis Address": f"rabbitmq host:{redis_host}"})

minio_host = os.environ.get('minio_host')

if minio_host == None:
    client = Minio("localhost:9000", "dataignite", "dataignite", secure=False)
    dilogger.info({"Minio host": f"Minio host:localhost:9000"})
else:
    client = Minio(minio_host, "dataignite", "dataignite", secure=False)
    dilogger.info({"Minio host": f"Minio host:{minio_host}"})

class Randomforest():
    def __init__(self):
            self.report = []
            self.credentials = pika.PlainCredentials('guest', 'guest')
            mqbroker = os.environ.get('mqbroker')
            if mqbroker == None:
                self.param = pika.ConnectionParameters('localhost', 5672, '/', self.credentials)
            else:
                self.param = pika.ConnectionParameters(mqbroker, 5672, '/', self.credentials)
            dilogger.info({sys._getframe().f_code.co_name: f"MQ Broker:{mqbroker}"})

            self.sub_connection = pika.BlockingConnection(self.param)
            self.sub_channel = self.sub_connection.channel()
            q_name = 'RandomforestClassifier_Call'
            mq = self.sub_channel.queue_declare(queue=q_name).method.queue
            self.sub_channel.basic_consume(mq, on_message_callback=self.wait_mq_signal, auto_ack=True)
            self.sub_channel.start_consuming()

    def wait_mq_signal(self,ch, method, properties, body):
        try:
            body = body.decode('utf-8')
            msg_json = json.loads(body)
            dilogger.info({sys._getframe().f_code.co_name: f"randomforest-subscribe:{msg_json}"})
            fit_param = msg_json['fit_param']
            if fit_param is not None:
                dilogger.info({"status": "ok", sys._getframe().f_code.co_name: "fit start"})
                fit_parameter={
                    'session_id': msg_json['session_id'],
                    'timestamp': msg_json['timestamp'],
                    'model_name': msg_json['model_name'],
                    'bucket_name': msg_json['bucket_name'],
                    'input_path': msg_json['input'],
                    'x_name_list': fit_param['x_name_list'],
                    'y_name': fit_param['target_name'],
                    'numbers_of_tree': fit_param['numbers_of_tree'],
                    'criterion': fit_param['criterion'],
                    'max_dept': fit_param['max_dept'],
                    'max_features': fit_param['max_features'],
                    'max_leaf_nodes': fit_param['max_leaf_nodes'],
                    'min_samples_leaf': fit_param['min_samples_leaf'],
                    'min_samples_split': fit_param['min_samples_split'],
                    'min_weight_fraction': fit_param['min_weight_fraction'],
                    'n_estimators': fit_param['n_estimators'],
                    'test_size': fit_param['test_size']
                }

                dilogger.info({sys._getframe().f_code.co_name: f"fit argument:{fit_parameter}"})
                self.fit(**fit_parameter)
                dilogger.info({sys._getframe().f_code.co_name: f"rfc process finished wait other request"})

        except Exception as e:
            session_id = fit_parameter['session_id']
            key_status = f'rfc_{session_id}'
            bjson_value = {
                            'status': 'error',
                            'session_id': str(msg_json['session_id']),
                            'timestamp': msg_json['timestamp'],
                            'msg': str(e),
                            'name': msg_json['model_name'],
                            'algorithm': 'rfc'
                           }
            r.set(key_status, json.dumps(bjson_value, ensure_ascii=False).encode('utf-8'))
            dilogger.error({"status": "error", sys._getframe().f_code.co_name: f"fit error:{bjson_value}"})

    def fit(self, **kargs):
        session_id = kargs['session_id']
        timestamp = kargs['timestamp']
        model_name = kargs['model_name']
        bucket_name = kargs['bucket_name']
        input_path = kargs['input_path']
        x_name_list = kargs['x_name_list']
        y_name = kargs['y_name']
        numbers_of_tree = kargs['numbers_of_tree']
        criterion = kargs['criterion']
        max_dept = kargs['max_dept']
        max_features = kargs['max_features']
        max_leaf_nodes = kargs['max_leaf_nodes']
        min_samples_leaf = kargs['min_samples_leaf']
        min_samples_split = kargs['min_samples_split']
        min_weight_fraction = kargs['min_weight_fraction']
        n_estimators = kargs['n_estimators']
        test_size = kargs['test_size']

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
                            'algorithm': 'rfc'}
                r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"xls data load start:{input_path}"})
                input_df = pd.read_excel(object_data)

                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'xls data load success',
                            'name': model_name,
                            'filename': input_path,
                            'algorithm': 'rfc'}
                r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"xls data load success:{input_path}"})

            elif 'csv' in file_type:
                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'csv data load start',
                            'name': model_name,
                            'filename': input_path,
                            'algorithm': 'rfc'}
                r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"csv data load start:{input_path}"})

                object_str = str(object_data, 'utf-8')
                data = StringIO(object_str)
                input_df = pd.read_csv(data)

                msg_json = {'status': 'ready',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'msg': 'csv data load success',
                            'name': model_name,
                            'filename': input_path,
                            'algorithm': 'rfc'
                            }
                r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: f"csv data load success:{input_path}"})

            else:
                msg_json = {'status': 'error',
                            'session_id': str(session_id),
                            'timestamp': timestamp,
                            'name': model_name,
                            'filename': input_path,
                            'msg': 'unknown data type',
                            'algorithm': 'rfc'
                            }
                r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
                dilogger.debug({"status": "error", sys._getframe().f_code.co_name: f"unknown data type:{input_path}"})

                return 0
        except Exception as e:
            msg_json = {'status': 'error',
                        'session_id': str(session_id),
                        'timestamp': timestamp,
                        'name': model_name,
                        'filename': input_path,
                        'msg': str(e),
                        'algorithm': 'rfc'
                        }
            r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))
            dilogger.debug({"status": "error", sys._getframe().f_code.co_name: f"unknown data type:{input_path}"})

            return 0

        X = input_df.drop([y_name], axis=1)
        y = input_df[y_name]
        class_labels = list(input_df[y_name].unique())

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        X_train = X_train[x_name_list]
        X_test = X_test[x_name_list]

        # 3.Pit
        msg_json = {'status': 'start',
                    'session_id': str(session_id),
                    'timestamp': timestamp,
                    'msg': 'model fit in process',
                    'name': model_name,
                    'filename': input_path,
                    'algorithm': 'rfc'
                    }
        r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))


        rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features=max_features, max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=n_estimators,
                               n_jobs=None, oob_score=False, random_state=0, verbose=0,
                               warm_start=False)
        rfc.fit(X_train, y_train)

        msg_json = {'status': 'finish',
                    'session_id': str(session_id),
                    'timestamp': timestamp,
                    'msg': 'model fit finish',
                    'name': model_name,
                    'filename': input_path,
                    'algorithm': 'rfc'
                    }
        r.set(f'rfc_{session_id}', json.dumps(msg_json, ensure_ascii=False).encode('utf-8'))

        y_pred = rfc.predict(X_test)
        dilogger.debug({"status": "ok", sys._getframe().f_code.co_name:'predict for X test'})

        accuracy = rfc.score(X_test, y_test)
        dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: 'score for X test'})

        msg_json = {'status': 'success',
                    'session_id': str(session_id),
                    'timestamp': timestamp,
                    'msg': 'model fit finish',
                    'name': model_name,
                    'filename': input_path,
                    'features': x_name_list,
                    'y_test': y_test,
                    'X_test': X_test,
                    'y_pred': y_pred,
                    'accuracy': accuracy,
                    'class_lbl': class_labels,
                    'model': rfc,
                    'algorithm': 'rfc'
                    }
        try:
            r.set(f'rfc_report_{session_id}', pickle.dumps(msg_json))
        except Exception as e:
            dilogger.debug({"status": "ok", sys._getframe().f_code.co_name: str(e) + ' model save error'})

if __name__ == '__main__':
    rfr = Randomforest()

