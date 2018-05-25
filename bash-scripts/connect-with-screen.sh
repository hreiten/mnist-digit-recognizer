source params.sh
ssh -t $user@$ip "cd $path ; bash" 'command; screen -S run ; bash' 'command; source ~/.bashrc'
