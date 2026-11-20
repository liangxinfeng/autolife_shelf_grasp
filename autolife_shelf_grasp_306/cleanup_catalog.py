"""One atomic document for enabled disposal categories and operator-taught appearances."""
import json
import numpy as np
from .core import atomic_json,unit
from .grounded_sam2 import normalize_prompt

DEFAULT_TYPES=('纸团','纸巾团','零食袋','塑料包装')


def checked_types(names,samples=()):
    if not isinstance(names,(tuple,list)):raise ValueError('垃圾种类应为列表')
    phrases={}
    for name in names:
        if not isinstance(name,str):raise ValueError('垃圾种类应为物品名称')
        name=name.strip()
        if not name:continue
        display,sep,query=name.partition('=')
        if not display.strip() or (sep and not query.strip()):raise ValueError('请输入中文名 = 英文识别词')
        try:phrase=normalize_prompt(query.strip() if sep else name).rstrip('.')
        except ValueError:
            if name in samples:continue
            raise
        if not phrase or '. ' in phrase or len(phrase)>55:raise ValueError('每行填写一种简短的物品名称')
        phrases.setdefault(phrase,name)
    return phrases


def library(root):
    path=root/'data/cleanup_types.json'
    data=json.loads(path.read_text('utf8')) if path.exists() else list(DEFAULT_TYPES)
    if isinstance(data,list):data=dict(version=1,types=data,samples={})
    if data.get('version')!=1:raise ValueError('垃圾清单版本不支持')
    return data


def load_types(root):
    return library(root)['types']


def save_types(root,names,open_waste=None):
    data=library(root)
    names=list(dict.fromkeys(n.strip() for n in names if n.strip()))
    checked_types(names,data['samples'])
    data.update(types=names,samples={k:v for k,v in data['samples'].items() if k in names})
    if open_waste is not None:data['open_waste']=bool(open_waste)
    atomic_json(root/'data/cleanup_types.json',data)


def enroll(root,name,item):
    name=name.strip()
    if not 1<=len(name)<=60 or '\n' in name:raise ValueError('请输入1～60字的垃圾种类名称')
    data=library(root)
    feature=np.asarray(item['feature'],float)
    if feature.ndim!=1 or not np.isfinite(feature).all():raise ValueError('样本外观特征无效')
    try:hint=normalize_prompt(item.get('suggested_name','')).rstrip('.')
    except ValueError:hint=''
    if not hint or len(hint)>55 or '. ' in hint:hint='object'
    if name not in data['types']:data['types'].append(name)
    examples=data['samples'].setdefault(name,[])
    example=dict(feature=unit(feature).tolist(),hint=hint)
    if 'views' in item:example['views']=[unit(v).tolist() for v in item['views']]
    if item.get('profile'):example['profile']=item['profile']
    examples.append(example)
    atomic_json(root/'data/cleanup_types.json',data)
    return len(examples)
