






#define BUFSIZE 30

int main(int argc, char **argv){

	int serv_sock;
	struct sockaddr_in serv_addr;

	fd_set reads, temps;
	int fd_max;

	char message[BUFSIZE];
	int str_len;
	struct timeval timeout;

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
	
	if(listen(serv_sock, 6) == -1)
		error_handling("listen() error");

	//---------------------------------------------------------------------
	/*now we will deal with select() and timeout*/

	FD_ZERO(&reads);
	FD_SET(serv_sock, &reads);

	while(1)
	{
		temps = reads;
		timeout.tv_sec = 5;
		timeout.tv_usec = 0;

		result = se

	}
	return 0;
}
