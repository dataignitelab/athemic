from minio import Minio
import os
import redis
import streamlit as st

redis_mgr = None
redis_host = os.environ.get('redis_host')
try:
    if redis_host == None:
        redis_mgr = redis.StrictRedis(host='localhost', charset="utf-8",port=6379, db=0)
    else:
        redis_mgr = redis.StrictRedis(host=redis_host, charset="utf-8",port=6379, db=0)
except Exception as e:
    st.error('Redis connection fail')
    st.error(str(e))