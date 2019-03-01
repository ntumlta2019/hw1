# Machine Learning HW1 教學  


<h3 id="basic">[基礎篇] PTT 爬蟲實際演練：</h3>
用 PTT 的電影版文章作為我們的爬蟲目標囉！

```python
import requests

def fetch(url):
    response = requests.get(url)
    response = requests.get(url, cookies={'over18': '1'})  # 一直向 server 回答滿 18 歲了 !
    return response

url = 'https://www.ptt.cc/bbs/movie/index.html'
resp = fetch(url)  # step-1

print(resp.text) # result of setp-1
```