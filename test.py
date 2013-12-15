from reporter import Reporter

test = 'http://www.washingtonpost.com/world/in-latvia-young-people-discover-new-passions-in-bad-economic-times/2013/07/29/ac638cac-efbf-11e2-8c36-0e868255a989_story.html'
test2 = 'http://blogs.hbr.org/2013/11/making-decisions-together-when-you-dont-agree-on-whats-important/'

my_reporter = Reporter()
my_reporter.read(url=test2)
print my_reporter.report_news()
