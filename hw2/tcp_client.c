#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define BUFSIZE 1024

void error_handling(char * message);

int main(int argc, char ** argv){
	int sock;
	char message[BUFSIZE];
	const char* message1;
	const char* message2;
	const char* message3;

	int str_len;
	struct sockaddr_in serv_addr;

	if(argc != 3){
		printf("Usage : %s <IP> <port>\n", argv[0]);
		exit(1);
	}

	sock = socket(PF_INET, SOCK_STREAM, 0);
	if (sock == -1)
		error_handling("socket() error");
	
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
	serv_addr.sin_port = htons(atoi(argv[2]));
	if(connect(sock, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) == -1)
		error_handling("connect() error!");
	
		/*message input & send*/
		// fputs("Enter a message to send(q to quit) : ", stdout);
		// fgets(message, BUFSIZE, stdin);
	

		message1 = "Handong Global University \nIs the best in the world\n|";
		message2 = "Today is Festival \nLet's all go out and play!\n|";
		message3 = "Computer Network \nEssential lecture\n|";
		//if(!strcmp(message1, "q\n"))
		write(sock, message1, strlen(message1));

		//if(!strcmp(message2, "q\n")) 
		write(sock, message2, strlen(message2));

		//if(!strcmp(message3, "q\n")) 
		write(sock, message3, strlen(message3));
	

		/*print messages received*/
		str_len = read(sock, message, BUFSIZE-1);
		message[str_len] = 0;
		printf("A message from the server : \n%s \n", message);
	

	close(sock);
	return 0;
	}


void error_handling(char *message){
	fputs(message, stderr);
	fputc('\n', stderr);
	perror("에러내용은: ");
	exit(1);
}
