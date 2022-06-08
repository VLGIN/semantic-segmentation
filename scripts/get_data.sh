mkdir -p assert/data
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=khanhbk20&password=Khanhbk20@123&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P assert/data
unzip assert/data/gtFine_trainvaltest.zip -d assets/data/
find assert/data -maxdepth 1 -type f -delete
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P assert/data
unzip assert/data/leftImg8bit_trainvaltest.zip -d assets/data/
find assert/data -maxdepth 1 -type f -delete
rm cookies.txt
rm index.html