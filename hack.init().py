import scrapy
import time as t
from faker import Factory
from wangyi.items import WangyiItem
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from scrapy.http import HtmlResponse
from random import choice
import re


headers = {
	'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
	'Accept-Encoding':'gzip,deflate,sdch',
	'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
	'Connection':'keep-alive',
	'Host':'music.163.com',
	'Upgrade-Insecure-Request':1,
	'User-Argent':f.user_agent()
}
name = 'wangyi'
start_urls = ['http://music.163.com/discover/playlist']

def parseTag(self,response):
	tagclass = response.xpath('//div[@class="bd"]/dl[@class="f-cb"]')
	tagname = tagclass[1].xpath('.//dd/a/text()').extract()
	baseurl = 'http://music.163.com/discover/playlist/?order=hot&limit=35&cat='
	specialtag = ['乡村'，'英伦'，'古风']
	for tag in specialtag:
		url = baseurl + tag
		yield scrapy.Request(url = url,headers = self.headers , callback = self.parsePage,meta = {'style':tag})


def parsePageCount(self,response):
	try:
		style = response.meta['style']
		pageCount = response.xpath('//div[@class="u-page"]/a')[-2].xpath('./text()').extract_first()
		baseurl = 'http://music.163.com/discover/playlist/?order=hot&limit=35&cat=' + style + '&offset='
		for i in range(5):
			url = baseurl + str(35 * i)
			yield scrapy.Request(url = url,headers = self.headers , callback = self.parsePage,meta = {'style':style})
	except:
		pass

def parsePage(self,response):
	try:
		style = response.meta['style']
		info = response.xpath('//div[@class="u-cover u-cover-1"]/a//@href').extract()
		for every in info:
			url = response.urljoin(every)
			yield scrapy.Request(url = url,headers = self.headers , callback = self.parseMusiclist,meta = {'style':style})
	except:
		pass

def parseMusiclist(self,response):
	style = response.meta['style']
	try:
		name = response.xpath('//div[@class="tit"]/h2/text()').extract_first()
	except:
		name = 'null'
	try:
		counts = response.xpath('//div[@id="content-operation"]/a')[2].xpath('.//i/text()').extract_first()
		counts = counts[1:-1]
	except:
		counts = -1
	meta = {'style':style, 'name' :name,'counts' : counts}
	musiclist = response.xpath('//ul[@class="f-hide"]/li/a/@href').extract();
	for music in musiclist:
		url  = reponse.urljoin(music)
		meta['url'] = url 
		yield scrapy.Request(url = url,headers = self.headers , callback = self.parseMusic,meta = meta)

def __init__(self):
	self.totalmusic = 0
	self.totallist = 0
	
	
 























