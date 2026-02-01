unset proxy_https 
unset proxy_http
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset ftp_proxy
unset rsync_proxy
unset no_proxy
git add *
git commit -m update_files
git push origin master
echo "Files updated and pushed to remote repository."