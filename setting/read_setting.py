import torch
import pandas as pd
import os
from easydict import EasyDict
from argparse import ArgumentParser
import argparse
def read_excel(loc,sheet_name):
    excel=pd.read_excel(loc,sheet_name=sheet_name)
    if excel.empty:
       raise 'trains are finished'
    set_pd=excel.head(1)
    set_dict=pd.DataFrame.to_dict(set_pd)
    set_dict=transform_type(set_dict)
    return set_dict
def begin_excel(loc,sheet_name):
    excel = pd.read_excel(loc,sheet_name=sheet_name)
    assert excel.empty!=True ,'Input is Empty!!!!!!!!!!!!!'
    dir_name=os.path.dirname(loc)
    writer=pd.ExcelWriter(os.path.join(dir_name,'save.xlsx'))
    excel.to_excel(writer,sheet_name='save',index=False)
    writer._save()
    writer.close()
def finish_excel(loc,sheet_name):
    #获得路
    dir_name = os.path.dirname(loc)
    #
    excel = pd.read_excel(loc, sheet_name=sheet_name)
    finishes_excel=pd.read_excel(os.path.join(dir_name,'finish.xlsx'))
    content=excel.head(1)
    finishes_excel=finishes_excel._append(content)
    writer=pd.ExcelWriter(os.path.join(dir_name,'finish.xlsx'))
    finishes_excel.to_excel(writer, sheet_name='finish', index=False)
    writer._save()
    writer.close()
    #
    excel = excel.drop(0)
    writer = pd.ExcelWriter(os.path.join(loc))
    excel.to_excel(writer, sheet_name=sheet_name, index=False)
    writer._save()
    writer.close()
def transform_type(datas):
    for key in datas.keys():
        if key=='mosaic_scale' or key=='mixup_scale':
            for i in datas[key].keys():
                tmp_data=datas[key][i]
                split=tmp_data.split('t')
                datas[key][i]=(float(split[0]),float(split[1]))
    return datas
def generate_args(args:ArgumentParser,set_dict:dict,is_read_excel:bool):
    if is_read_excel:
        assert len(set_dict)>0,'names and setting is empty!!!!please check the path of loc!!!'
    configs = vars(args)
    for name,setting in set_dict.items():
        assert name in configs.keys(),'{} can not be found! keys error! Please reset the excel!!!'.format(name)
        configs[name]=setting[0]
    args=argparse.Namespace(**configs)
    return args
