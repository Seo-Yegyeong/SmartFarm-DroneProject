#define BUFSIZE 30
void error_handling(char *message);

int main(int argc, char ** argv){
	int sock;
	char message[BUFSIZE];
	int str_len, addr_size, i;

	char MSG1[] = "Good ";
	char MSG2[] = "Evening ";
	char MSG3[] = "Everybody!";

	struct sockaddr_in serv_addr;
	struct sockaddr_in from_addr;

	if(argc != 3){
		printf("Usage : %s <IP> <port>\n", argv[0]);
		exit(1);
	}

	sock = socket(PF_INET, SOCK_DGRAM, 0);
	if(sock == -1)
		error_handling("UDP socket() error");
	
	memset(&serv_addr, 0, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
	serv_addr.sin_port = htons(atoi(argv[2]));

	for(i=0 ; i<3 ; i++){
		addr_size = sizeof(from_addr);
		str_len = recvfrom(sock, message, BUFSIZE, 0, (struct sockaddr*) &from_addr, &addr_size);
		message[str_len]='\0';
		printf("The message %d from the server: %s \n", i, message);
	}
	close(sock);
	return 0;
}
