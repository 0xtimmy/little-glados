import re

test_string = "please <write to=\"hello.txt\"> ooga booga boo </write> thanks!"

regex = re.compile(r"<write\s+to=\"(?P<filename>[^\"]+)\"\s*>(?P<content>(?:.|\n)+(?=<\/write>))<\/write>")

match = regex.search(test_string)

print (f"filename: {match.group('filename')}")
print (f"content: {match.group('content')}")