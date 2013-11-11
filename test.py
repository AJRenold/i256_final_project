from reporter import Reporter

my_reporter = Reporter()
my_reporter.read(url='http://paper.li/totaltelecom/1296049118')
print my_reporter.report_news()
