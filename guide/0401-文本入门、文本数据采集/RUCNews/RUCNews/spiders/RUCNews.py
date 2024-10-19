import scrapy


class RUCNewsSpider(scrapy.Spider):
    name = "RUC"

    def start_requests(self):
        urls = [
            'http://news.ruc.edu.cn/archives/275767/',
            'http://news.ruc.edu.cn/archives/276221/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = 'RUCNews-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)