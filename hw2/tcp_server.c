#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define BUFSIZE 1024

void error_handling(char *message);

int main(int argc, char **argv){
	int serv_sock;
	int clnt_sock;
	char message[BUFSIZE];
	
	int str_len;
	char* parseMsg;
	char* temp;
	char* parseLine;
	int i;

	struct sockaddr_in serv_addr;
	struct sockaddr_in clnt_addr;
	int clnt_addr_size;

	if(argc!=2){
		printf("Usage : %s <port>\n", argv[0]);
		exit(1);
	}

	serv_sock = socket(PF_INET, SOCK_STREAM, 0);
	if(serv_sock == -1)
		error_handling("socket() error");
	
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(atoi(argv[1]));

	if(bind(serv_sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) == -1)
		error_handling("bind() error");

	if(listen(serv_sock, 10) == -1)
		error_handling("listen() error");
	
	clnt_addr_size = sizeof(clnt_addr);
	clnt_sock = accept(serv_sock, (struct sockaddr*) &clnt_addr, &clnt_addr_size);
	if(clnt_sock == -1)
		error_handling("accept() error");

	
	sleep(5);
	str_len = read(clnt_sock, message, BUFSIZE);
	write(clnt_sock, message, str_len);


	parseMsg = strtok(message, "|");

	i=0;
	while(parseMsg != NULL){
		i++;
		printf("Received message %d:\nMSG%d: ", i,i);
		printf("%s\n", parseMsg);
		parseMsg = strtok(NULL, "|");
	}


	close(clnt_sock);
	close(serv_sock);
	return 0;
}

void error_handling(char *message){
	fputs(message, stderr);
	fputc('\n', stderr);
	exit(1);
}
