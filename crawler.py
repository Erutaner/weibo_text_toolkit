import os
import queue
import calendar
import csv
import threading
from datetime import datetime, timedelta
from urllib.parse import  quote,unquote
from time import sleep
from lxml import etree
import requests
class searsh(object):
    ip_queue = queue.Queue() #放ip的队列
    num=0
    word_list = ['疫情放开']
    num_list= queue.Queue()
    url_list=[]
    csv_list = []#用于存放当前关键词的csv所存放的起始时间和数据
    aip_list='http://route.xiongmaodaili.com/xiongmao-web/api'
    lock = threading.Lock()
    def __init__(self,cookies):
        self.cookies = cookies
        pass

    # 用于判断某个月中有多少天
    def month_days_mum(self,year, month):
        return calendar.monthrange(year, month)[1]

    #获取到ip的函数
    def getIp_list(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
        }
        url = self.aip_list
        while True:
            try:
                response = requests.get(url=url, headers=headers, timeout=10).text
                break
            except:
                sleep(1)
                continue
        res = response.split('\r\n')

        print(res)
        for i in res:
            self.ip_queue.put({"https":'https://'+i})
        print('当前ip',self.ip_queue.qsize())


    #拿到单个ip
    def getIp(self):
        print(self.ip_queue.qsize())
        if self.ip_queue.qsize()<5:
                self.getIp_list()
        return self.ip_queue.get()


    #进行数据请求的函数
    def request_url(self,url, ip):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
        }
        n = 0
        w = 0
        while True:
            try:
                # url='https://www.baidu.com'
                # print(self.cookies)
                res = requests.get(url=url, headers=headers, timeout=10, proxies=ip, cookies=self.cookies)
                if 'Sina Visitor System' in res.text or '����ͨ��֤' in res.text:
                    sleep(120)
                    print('请在txt中更新cookie')
                    with open('cookie.txt', 'r', encoding='utf-8') as f:
                        a = f.read()
                    self.cookies = {
                        "cookie": a
                    }
                    continue
                break
            except Exception as e:
                print(e)
                if w < 3:
                    w = w + 1
                    continue
                ip = self.getIp()
                n = n + 1
                continue
        return {'res': res, 'ip': ip}

    # 用于判断某个月中有多少天

    def month_days_mum(self,year, month):
        return calendar.monthrange(year, month)[1]

    # 输出 按小时的list
    def time_list(self,min, max):
        min_time  = min.split(' ')[1]
        max_time = max.split(' ')[1]
        min  =min.split(' ')[0]
        max = max.split(' ')[0]
        all_list = []
        min = min.split('-')
        max = max.split('-')
        min_year = int(min[0])
        min_mounth = int(min[1])
        min_day = int(min[2])
        min_time = int(min_time)
        while True:
            if min_year * 360 + min_mounth * 30 + min_day <=int(max[0]) * 360 + int(max[1]) * 30 + int(max[2]) and int(min_time)<=int(max_time):
                while min_time <= 24:
                    if  min_year * 360 + min_mounth * 30 + min_day >=int(max[0]) * 360 + int(max[1]) * 30 + int(max[2]) and int(min_time)>int(max_time):
                            break
                    if min_mounth < 10:
                        mouth = '0' + str(min_mounth)
                    else:
                        mouth = str(min_mounth)
                    if min_day < 10:

                        all_list.append(
                            str(min_year) + '-' + mouth + '-' + '0' + str(min_day) + '-' + str(min_time))
                    else:
                        all_list.append(
                            str(min_year) + '-' + mouth + '-' + str(min_day) + '-' + str(min_time))
                    min_time = min_time + 1
                if min_day == self.month_days_mum(min_year, min_mounth):
                    min_mounth = min_mounth + 1
                    if min_mounth == 13:
                        min_mounth = 1
                        min_year = min_year + 1
                    min_day = 1
                else:
                    min_day = min_day + 1
                min_time = 0
            else:
                break
        return all_list


    def changeDate(self,date):  # 用于改编时间
        return datetime.strptime(date, "%a %b %d %H:%M:%S +0800 %Y")
     # 解析文本
    def jx_text(self,tree,ip):

        for li in tree.xpath('//div[@action-type="feed_list_item"]'):
            try:
                name = li.xpath('.//a[@class="name"]/text()')[0]  # 用户名
                date = li.xpath('.//div[@class="from"]/a/text()')[0].strip()  # 时间
                if '年' not in date:
                    date = '2023年'+date
                # 定义输入字符串的格式
                input_format = "%Y年%m月%d日 %H:%M"
                # 将输入字符串转换为 datetime 对象
                date = str(datetime.strptime(date, input_format))
                cbox = li.xpath('.//p[@node-type="feed_list_content_full"]')  # 内容
                cbox = li.xpath('.//p[@node-type="feed_list_content"]')[0] if not cbox else cbox[0]

                cont = ''.join(cbox.xpath('.//text()')).strip()
                tran = li.xpath('.//div[@class="card-act"]/ul/li[1]/a//text()')[1].strip()  # 转发
                try:
                    tran = eval(tran)
                except:
                    tran = 0
                comm = li.xpath('.//div[@class="card-act"]/ul/li[2]/a//text()')[0].strip()  # 评论
                try:
                    comm = eval(comm)
                except:
                    comm = 0
                like = li.xpath('.//div[@class="card-act"]/ul/li[3]/a/button/span[2]/text()')[0].strip()  # 点赞
                try:
                    like = eval(like)
                except:
                    like = 0
                bid = li.xpath('.//div[@class="from"]/a/@href')[0].split('?')[0].split('/')[4]  # 文章mid
                sork = 0
                pos = ''
                #while sork<=3:
                #    try:
                #        url = f"https://weibo.com/ajax/statuses/show?id={bid}&locale=zh-CN"
                #        res = self.request_url(url,ip)
                #        data = res['res'].json()
                #        date = str(self.changeDate(data['created_at']))
                #        ip = res['ip']
                #        try:
                #            pos = data['region_name'].split(' ')[1]
                #            # ip = res['ip']
                #        except:
                #            break
                #        break
                #    except:
                #        sork+=1
                #        ip = self.getIp()
                #        continue
                self.lock.acquire()
                file = self.judge_time_return_csv(date)
                file.writerow([name,date,cont,tran,comm,like,pos])
                print(name,date,cont,tran,comm,like,pos)
                self.lock.release()
                self.num += 1
                print('当前采集数',self.num)
                if self.ip_queue.qsize() < 5:
                    self.getIp_list()
            except Exception as e:
                print(e)
                continue
        return ip

    def main(self,name):
        ip = self.getIp()
        while self.num_list.qsize()>0:
            print('当前采集数为',self.num)
            print('当前队列剩余元素为：', self.ip_queue.qsize())
            flag = self.num_list.get()
            page = 1
            while page<=50:
                url = 'https://s.weibo.com/weibo?q=' + quote(name) + '&scope=ori&suball=1&timescope=custom%3A' + \
                      self.url_list[
                          flag] + '%3A' + self.url_list[flag + 1] + '&Refer=g&page='+str(page)
                print(url)
                res0 = self.request_url(url, ip)
                res = res0['res']
                ip = res0['ip']
                # print(res.text)
                if '抱歉，未找到' in res.text:
                    break
                tree = etree.HTML(res.text)
                ip=self.jx_text(tree,ip)
                page = page+1
            flag = flag + 1
            print('当前：',name, self.url_list[flag])


    #按4个小时为一模块进行分割，返回字典
    def split_time(self, start_time, end_time, word):
        current_time = start_time
        data = []
        while current_time < end_time:
            f = open('疫情放开//'+word + str(current_time).replace(':', '-') + '.csv', 'a', newline='', encoding='utf-8-sig')
            f = csv.writer(f)
            f.writerow(['用户名','时间','微博正文','转发数','评论数','点赞数','ip'])
            data.append( {
                'start_time': current_time,
                'end_time': current_time + timedelta(hours=24),
                'csv_name': f
            })
            current_time += timedelta(hours=24)
        return data
    #返回该微博时间所处的时间段的微博所进行存储的csv
    def judge_time_return_csv(self,time):
        t = time
        time  =datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        # print(self.csv_list)
        for d in self.csv_list:
            if d['start_time']<=time<=d['end_time']:
                break
        return d['csv_name']
    #开启线程
    def start(self,name):
        flag = 20
        threads = []
        for i in range(flag):
            t = threading.Thread(target=self.main, args=(name,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()

    #基础的初始设置
    def start_p(self):
        global writer
        self.getIp_list()
        for word in self.word_list:
            self.url_list = self.time_list('2022-12-1 0', '2023-8-1 0')
            start = datetime.strptime('2022-12-01 00:00:00', "%Y-%m-%d %H:%M:%S")  # 开始时间
            end =  datetime.strptime('2023-08-01 00:00:00', "%Y-%m-%d %H:%M:%S")  # 结束时间
            self.csv_list = self.split_time(start,end,word)
            for a in range(len(self.url_list)):
                self.num_list.put(a)
            print(self.num_list.qsize())
            self.start(word)

if __name__=='__main__':
    with open('cookie.txt', 'r', encoding='utf-8') as f:
        a = f.read()
    cookies = {
        "cookie": a
    }
    s = searsh(cookies)
    isExists = os.path.exists('疫情放开')
    # 判断结果
    if not isExists:  # 查看是否有数据这个文件夹
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs('疫情放开')
    s.start_p()




