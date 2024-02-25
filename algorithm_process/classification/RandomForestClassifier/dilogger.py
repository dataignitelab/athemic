from elasticsearch import Elasticsearch
import logging
from datetime import datetime
import sys
class DIlogger:

    def __init__(self, address_list, index, level):
        logPath = './'
        filename = 'statistics'
        if address_list is not None:
            self.es = Elasticsearch(address_list)
        else:
            self.es = None
        self.index = index

        self.logger = logging.getLogger(index)
        if level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif  level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level == "ERROR":
            self.logger.setLevel(logging.ERROR)

        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, filename), encoding='utf-8')
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        fileHandler.setFormatter(logFormatter)
        self.logger.addHandler(fileHandler)

        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def debug(self, log_msg):
        msg_body = {
            "index": self.index,
            "msg": log_msg,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}
        if self.es is not None:
            try:
                self.es.index(index=self.index, body=msg_body)
            except Exception as e:
                self.es = None
        self.logger.debug(msg_body)

    def info(self, log_msg):
        msg_body = {"index": self.index,
                    "msg": log_msg,
                    "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}
        if self.es is not None:
            try:
                self.es.index(index=self.index, body=msg_body)
            except Exception as e:
                self.es = None
        self.logger.info(msg_body)

    def error(self, log_msg):
        msg_body = {
            "index": self.index,
            "msg": log_msg,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'}
        if self.es is not None:
            try:
                self.es.index(index=self.index, body=msg_body)
            except Exception as e:
                self.es = None
        self.logger.error(msg_body)