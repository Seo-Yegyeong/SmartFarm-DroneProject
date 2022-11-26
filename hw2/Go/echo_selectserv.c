#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/select.h>

#define BUF_SIZE 100
#define NICKNAME 30
void error_handling(char *buf);

int main(int argc, char **argv)
{
	int serv_sock;
  int clnt_sock;
	struct sockaddr_in serv_adr;
  struct sockaddr_in clnt_adr;
	struct timeval timeout;
	fd_set reads, cpy_reads;

	socklen_t adr_sz;
	int fd_max, str_len, fd_num, i, j, temp;
	char buf[BUF_SIZE];
  char realmsg[BUF_SIZE];
  char nickname[][NICKNAME]={0,};
  int flag[NICKNAME]={0,};
  
	if (argc != 2) {
		printf("Usage : %s <port>\n", argv[0]);
		exit(1);
	}

	serv_sock = socket(PF_INET, SOCK_STREAM, 0);
  if(serv_sock == -1)
    error_handling("socket() error");
    
	memset(&serv_adr, 0, sizeof(serv_adr));
	serv_adr.sin_family = AF_INET;
	serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_adr.sin_port = htons(atoi(argv[1]));
	
	if (bind(serv_sock, (struct sockaddr*) &serv_adr, sizeof(serv_adr)) == -1)
		error_handling("bind() error");
   
	if (listen(serv_sock, 5) == -1)
		error_handling("listen() error");

	FD_ZERO(&reads);
	FD_SET(serv_sock, &reads);
	fd_max = serv_sock;

	while (1)
	{
		cpy_reads = reads;
		timeout.tv_sec = 5;
		timeout.tv_usec = 5000;

		if ((fd_num = select(fd_max+1, &cpy_reads, 0, 0, &timeout)) == -1)
			break;
		
		if (fd_num == 0)
			continue;
      
    printf("\nfd_num == %d\n", fd_num); 

		for (i = 3; i < fd_max + 1; i++)
		{
			if (FD_ISSET(i, &cpy_reads))
			{
				if (i == serv_sock)     // connection request!
				{
					adr_sz = sizeof(clnt_adr);
					clnt_sock =
						accept(serv_sock, (struct sockaddr*)&clnt_adr, &adr_sz);
          if(clnt_sock == -1)
            error_handling("accept() error");
          
          printf("Test : serv_sock == %d, clnt_sock == %d\n", serv_sock, clnt_sock);
          printf("fd_max == %d\n",fd_max);
					FD_SET(clnt_sock, &reads);
					if (fd_max < clnt_sock)
						fd_max = clnt_sock;
					printf("connected client (i==%d): %d \n\n", i, clnt_sock);
          flag[clnt_sock] = 1;
				}
				else    // read message!
				{
					str_len = read(i, buf, BUF_SIZE);
					if (str_len == 0)    // close request!
					{
						FD_CLR(i, &reads);
						close(i);
						printf("closed client: %d \n", i);
					}
					else
					{
						//write(i, buf, str_len);    // echo!
            buf[str_len-1] = '\0';
            if(flag[clnt_sock] == 1){
              strcpy(nickname[clnt_sock], buf);
              sprintf(realmsg,"[%s has entered the chatroom!](Q to quit)\n",buf);
            }
            else{
              sprintf(realmsg,"[%s] %s\n",nickname[clnt_sock], buf);
            }
            
            for(j=4; j< fd_max+1 ; j++){
              if(i!=j)
                write(j, realmsg, strlen(realmsg));
            }
            if(flag[clnt_sock] == 1){
              write(i, realmsg, strlen(realmsg));
              for(j=4; j< fd_max+1 ; j++){
              
              
              //sprintf(realmsg, 
              }
              flag[clnt_sock] = 0;
            }
            printf("\nMessage : %s\n\n",realmsg);
					}
				}
        printf("i");
			}
      printf("t");
		}
   
     //sleep 시켰다가 이후에 연결된 clients가 없다면 종료!
	}
  printf("%s", buf);
	close(serv_sock);
	return 0;
}

void error_handling(char *buf)
{
	fputs(buf, stderr);
	fputc('\n', stderr);
	exit(1);
}