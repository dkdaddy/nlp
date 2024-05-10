# results = [
#     [.9, .1, "hello", True],
#     [.6, .4, "world", True],
#     [.55, .45, "world", True],
#     [.45, .55, "world", True], 
#     [.35, .65, "world", True], 
#     [.25, .75, "world", True], 
#     [.1, .9, "hello", False],
#     [.2, .8, "world", False],
#     [.4, .6, "world", False],
#     [.6, .4, "world", False]
# ]
def isTruePositive(entry):
    return entry[3] and entry[1]>=0.5
def isFalseNegative(entry):
    return entry[3] and entry[1]<0.5
def isFalsePositive(entry):
    return not entry[3] and entry[1]>=0.5
def isTrueNegative(entry):
    return not entry[3] and entry[1]<0.5

def classFromCategory(entry):
    if isTruePositive(entry):
        return 'tp'
    if isFalseNegative(entry):
        return 'fn'
    if isFalsePositive(entry):
        return 'fp'
    if isTrueNegative(entry):
        return 'tn'
    return None

ipsum = "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?"

def confusionMatrix(tp,fp,fn,tn):
    total = tp+fp+fn+tn
    accuracy = (tp+tn)/total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    html = f"<table><tr><td>{accuracy=:.3f}</td><td>{precision=:.3f}</td><td>{recall=:.3f}</td></tr></table>"
    html += f"<table><tr><td class='tp'>TP = {tp}</td><td class='fp'>FP = {fp}</td></tr><tr><td class='fn'>FN = {fn}</td><td class='tn'>TN = {tn}</td></tr></table>\n"
    return html

def html_table(rows):
    html = "<table>\n"
    for row in rows:
        html += "<tr>"
        pos, neg, text, expected = row
        # text = ipsum
        cssClassPos = expected and classFromCategory(row)
        cssClassNeg = not expected and classFromCategory(row)
        html += f"<td class='{cssClassNeg}'>{pos:.3f}</td><td class='{cssClassPos}'>{neg:.3f}</td><td><div class='text'>{text}</div></td>"
        html += "</tr>\n"
    html += "</table>\n"
    return html
style = """
<style>
body {background-color: powderblue;}
table, th, td {
  border: 1px solid;
  border-collapse: collapse;
}
table {
    margin : 3pt;
}
td {
    margin : 5pt;
    padding: 4pt;
}
.text {
  display: inline-block;
  max-width: 400pt;
  vertical-align: bottom;
  overflow: hidden;
  white-space: nowrap;
}
.text:hover{
  max-width: 100%;
  white-space: wrap;
}
.tp, .tn {
    background-color: rgb(10,243,33);
}
.fp, .fn {
    background-color: rgb(244,123,133);
}
</style>
"""

def get_html(results):
    
    tp = sum(1 for x in results if isTruePositive(x))
    fp = sum(1 for x in results if isFalsePositive(x))
    fn = sum(1 for x in results if isFalseNegative(x))
    tn = sum(1 for x in results if isTrueNegative(x))
    cm = confusionMatrix(tp, fp, fn, tn)
    table = html_table(results)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    {style}
    </head>
    <body>
    <div>
    {cm}
    </div>
    <div>
    {table}
    </div>
    </body>
    </html>
    """
    return html

# array of [neg_prob, pos_prob, text, true pos/neg]
def format_page(data):
    results = data
    f = open("tmp.html", "w")
    html = get_html(results)
    f.write(html)
    f.close()
