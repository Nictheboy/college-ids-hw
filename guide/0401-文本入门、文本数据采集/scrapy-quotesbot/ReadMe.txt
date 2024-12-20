Running the spiders

You can run a spider using the scrapy crawl command, such as:

	start Anaconda Prompt
	d:
	cd D:\2021-07-18《数据科学概论》new plan\2022newPPT\0401-文本入门、文本数据采集\scrapy-quotesbot
	scrapy crawl toscrape-css


If you want to save the scraped data to a file, you can pass the -o option:
	start Anaconda Prompt
	d:
	cd D:\2021-07-18《数据科学概论》new plan\2022newPPT\0401-文本入门、文本数据采集\scrapy-quotesbot
	scrapy crawl toscrape-css -o quotes.json

	或者
	scrapy crawl toscrape-css -o quotes.csv -t csv
