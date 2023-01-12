import pdfkit

config = pdfkit.configuration(wkhtmltopdf='C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe')
pdfkit.from_url('https://www.apple.com/iphone/', 'out.pdf', configuration=config)
