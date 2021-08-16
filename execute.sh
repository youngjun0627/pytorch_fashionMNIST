#!/bin/sh

tmux rename-window user
tmux new-session -d
tmux split-window -v
tmux select-pane -t 0
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

tmux send-keys -t 4 'conda activate uchanpython' C-m
tmux send-keys -t 4 'source ./client-py/my-backend-ai.sh' C-m
tmux send-keys -t 0 'conda activate uchanpython' C-m
tmux send-keys -t 0 'sh ./exec_manager.sh' C-m
tmux send-keys -t 2 'conda activate uchanpython' C-m
tmux send-keys -t 2 'sh ./exec_agent.sh' C-m
tmux send-keys -t 3 'conda activate uchanpython' C-m
tmux send-keys -t 3 'sh ./exec_storage.sh' C-m
tmux send-keys -t 1 'conda activate uchanpython' C-m
tmux send-keys -t 1 'sh ./exec_webserver.sh' C-m
tmux send-keys -t 4 'cd client-py' C-m
tmux select-pane -t 4
