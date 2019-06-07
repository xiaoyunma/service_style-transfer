import os
import uuid
import json
from flask import Flask, request, Response, \
    render_template, flash
from flask_sqlalchemy import SQLAlchemy
import config
from io import BytesIO
import base64
import numpy as np
import cv2

from fastTranfer import eval

app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)


# 创建数据库模型
class styleuser(db.Model):
    __tablename__ = 'User'
    UserID = db.Column(db.String, primary_key=True)
    UserName = db.Column(db.String(50), nullable=False)
    UserPhone = db.Column(db.String(50), nullable=False)
    UserPassword = db.Column(db.String(50), nullable=False)
    IsOnline = db.Column(db.Integer, nullable=False)
    FeedbackContent = db.Column(db.String(255),nullable=True)
    db.create_all()


# 注册服务
@app.route('/register', methods=['POST'])
def register():
    r = request.get_json()
    # 接收从客户端传过来的参数
    recvUserName = r['UserName']
    recvUserPhone = r['UserPhone']
    recvUserPassword = r['UserPassword']
    # 将数据集合赋给userinfo
    userinfo = styleuser(UserID=str(uuid.uuid1()), UserName=recvUserName, UserPhone=recvUserPhone,
                         UserPassword=recvUserPassword, IsOnline=0)
    # 将数据添加到数据库
    db.session.add(userinfo)
    # 提交事务
    db.session.commit()
    # 注册状态
    state = {'state': '1'}
    JsonRes = json.dumps(state)
    response = Response(JsonRes)
    response.headers['Content-type'] = 'application/json; charset=utf-8'
    response.headers['Content-Length'] = str(len(JsonRes))
    response.headers['Connection'] = 'close'
    print(response.headers)
    return response  # 将状态值返回给客户端


# 登录服务
@app.route('/login', methods=['POST'])
def login():
    r = request.get_json()
    # 接收从客户端传过来的登录信息
    recvUserPhone = r['UserPhone']
    recvUserPassword = r['UserPassword']
    # 将符合查询条件的信息赋给result
    result = styleuser.query.filter(styleuser.UserPhone == recvUserPhone,
                                    styleuser.UserPassword == recvUserPassword).first()
    # 初始化状态字典
    # failed当为“1”时，说明密码错误或者账号不存在；当为“0”时，继续判断是否已登录；
    # IsOnline当为“1”是，说明此账号已经登录，为“0”是，登录成功；将登录状态写入数据库；
    # 默认值都为“1”
    returnValue = {'failed': 1, 'IsOnline': '1'}
    # 如果查询结果为空，则failed=1，停止执行，将状态返回客户端
    if (result == None):
        returnValue['failed'] = 1
    else:
        # 当账号密码都正确时，继续判断此账号是否已经登录
        if (result.IsOnline == 0):
            returnValue['IsOnline'] = '0'
            result.IsOnline = 1
            db.session.commit()
        else:
            returnValue['IsOnline'] = '1'
        returnValue['failed'] = 0

    JsonRes = json.dumps(returnValue)
    response = Response(JsonRes)
    response.headers['Content-type'] = 'application/json; charset=utf-8'
    response.headers['Content-Length'] = str(len(JsonRes))
    response.headers['Connection'] = 'close'
    print(response.headers)
    return response


#反馈服务
@app.route('/feedback', methods=['POST'])
def feedback():
    r = request.get_json()
    # 接收从客户端传过来的参数
    recvFeedbackContent = r['FeedbackContent']
    # 将数据集合赋给userinfo
    results = styleuser.query.filter_by(styleuser.UserPhone=="15109607077").first()
    # 将数据添加到数据库
    #修改
    results.FeedbackContent=recvFeedbackContent
    # 提交事务
    db.session.commit()
    # 提交状态
    state = {'state': '1'}
    JsonRes = json.dumps(state)
    response = Response(JsonRes)
    response.headers['Content-type'] = 'application/json; charset=utf-8'
    response.headers['Content-Length'] = str(len(JsonRes))
    response.headers['Connection'] = 'close'
    print(response.headers)
    return response  # 将状态值返回给客户端
























# 风格迁移服务
@app.route('/test', methods=['POST'])
def test():
    styleDict = {'0': 'candy', '1': 'cubist', '2': 'feathers', '3': 'denoised_starry',
                 '4': 'udnie', '5': 'wave', '6': 'scream', '7': 'mosaic'}
    # 接收客户端的信息
    r = request.get_json()
    print(request.headers)
    recvCategory = r['category']
    recvBase64 = r['base64']
    recvStyleID = r['styleID']
    print(recvStyleID)
    bImg = base64.b64decode(bytes(recvBase64.encode('utf-8')))

    # 服务器处理从客户端接收到的图片字节流
    imgArray = cv2.imdecode(np.fromstring(bImg, np.uint8), 1)  # 字节流转为numpy数组
    imgHeight, imgWidth = imgArray.shape[0], imgArray.shape[1]
    if imgHeight > 1080:
        imgHeight = 1080
    if imgWidth > 600:
        imgWidth = 600
    image = cv2.resize(imgArray, (imgWidth, imgHeight), interpolation=cv2.INTER_CUBIC)  # 对图像数组进行裁剪
    bImg = cv2.imencode('.jpg', image)[1]  # numpy转换为字节流
    image_file = BytesIO()
    image_file.write(bImg)
    image_file.seek(0)
    print(type(image_file))
    resultBytes = eval.main(image_file, recvCategory, 'fastTranfer/models/' + styleDict[
        recvStyleID] + '.ckpt-done')  # os.system("activate sadad && python eval.py")
    # 返回给服务器
    base64_data = (base64.b64encode(resultBytes)).decode('utf-8')
    strImg = {"category": recvCategory, "base64": base64_data}
    JsonRes = json.dumps(strImg)
    response = Response(JsonRes)
    response.headers['Content-type'] = 'application/json; charset=utf-8'
    response.headers['Content-Length'] = str(len(JsonRes))
    response.headers['Connection'] = 'close'
    print(response.headers)
    return response


if __name__ == '__main__':
    app.run(host='192.168.43.233', threaded=False)
