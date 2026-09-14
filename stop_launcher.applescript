-- 关闭 Streamlit
try
	do shell script "pkill -f 'streamlit run web_app.py'"
end try

-- 关闭本地 PDF 服务
try
	do shell script "pkill -f 'http.server 8502'"
end try

display dialog "学术文献助手及 PDF 原文服务已关闭。" buttons {"好"} default button "好"
