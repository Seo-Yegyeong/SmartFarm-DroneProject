#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUF_SIZE 32
void error_handling(char *message);

int main(int argc, char *argv[])
{
	int sd;
	FILE *fp;
	
	char file_name[BUF_SIZE];
	char buf[BUF_SIZE];
	int read_cnt;
	int read_size;
	struct sockaddr_in serv_adr;
	if (argc != 4) {
		printf("Usage: %s <IP> <port> <file name> \n", argv[0]);
		exit(1);
	}
	
	fp = fopen(argv[3], "rb");
	sd = socket(PF_INET, SOCK_STREAM, 0);   

	memset(&serv_adr, 0, sizeof(serv_adr));
	serv_adr.sin_family = AF_INET;
	serv_adr.sin_addr.s_addr = inet_addr(argv[1]);
	serv_adr.sin_port = htons(atoi(argv[2]));

	connect(sd, (struct sockaddr*)&serv_adr, sizeof(serv_adr));
	
	// Send file name 
	strcpy(file_name, argv[3]);
	write(sd, file_name, BUF_SIZE);

	// Send file data 
	read_size = 0;
	while(1)
	{
		read_cnt = fread((void*)buf, 1, BUF_SIZE, fp);
		read_size += read_cnt;
		if (read_size % 1024 == 0)
			printf("Recv %d bytes \n", read_size);

		if (read_cnt < BUF_SIZE)
		{
			write(sd, buf, read_cnt);

			break;
		}
		write(sd, buf, BUF_SIZE);
	}
	
	printf("Recv %d bytes \n", read_size);
	shutdown(sd, SHUT_WR);	
	read(sd, buf, BUF_SIZE);
	printf("Message from server: %s \n", buf);

	fclose(fp);
	close(sd);
	return 0;
}

void error_handling(char *message)
{
	fputs(message, stderr);
	fputc('\n', stderr);
	exit(1);
}