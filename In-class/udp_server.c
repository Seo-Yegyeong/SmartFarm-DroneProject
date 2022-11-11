#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define BUFSIZE 300
void error_handling(char * message);

int main(int argc, char ** argv){
	int serv_sock;
	int BUFFSIZE = atoi(argv[2]);

	int str_len, num = 0;
	int i;
	char * parseMsg;
	
	char fmessage[BUFSIZE];
	char message[BUFSIZE];
	
	struct sockaddr_in serv_addr;
	struct sockaddr_in clnt_addr;
	int clnt_addr_size;
	

	if(argc!=3) {
		printf("Usage : %s <port>\n", argv[0]);
		exit(1);
	}

	serv_sock = socket(PF_INET, SOCK_DGRAM, 0);
	if(serv_sock == -1)
		error_handling("UDP socket() error");

	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(atoi(argv[1]));

	if(bind(serv_sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) == -1)
		error_handling("bind() error");
	
	while(1){

	for(i=0; i<3; i++){
              clnt_addr_size = sizeof(clnt_addr);
              
              str_len = recvfrom (serv_sock, message, BUFFSIZE, 0,
                                  (struct sockaddr*) &clnt_addr, &clnt_addr_size);
              message[str_len] = '\0';
              strncat(fmessage, message, str_len);
              strcat(fmessage,"|");
          	 // printf("message: %s\n", message);
              sendto(serv_sock, message, str_len, 0, (struct sockaddr*) &clnt_addr, sizeof(clnt_addr));
              
    }
	sleep(5);

	parseMsg = strtok(fmessage, "|");
	i=0;
	while(parseMsg != NULL){
		i++;
		printf("Received message %d: %s\n", i, parseMsg);
		parseMsg = strtok(NULL, "|");
	}
	}

	return 0;
}

void error_handling(char *message){
	fputs(message, stderr);
	fputc('\n', stderr);
	exit(1);
}
