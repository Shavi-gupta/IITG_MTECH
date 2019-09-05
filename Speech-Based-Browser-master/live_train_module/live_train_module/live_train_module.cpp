
// live_test_model.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "iostream"
#include "fstream"
#include "string"
#include "math.h"
#include "iomanip"

#define CODE_BOOK_SIZE 32
#define FRAME_SIZE 320
#define FRAMES 130
#define N 5
#define M 32
#define T 130
#define WORDS 13
#define ITERATIONS 3

using namespace std;

ifstream in, in1,in_obs;
ofstream out, out1;

char words[13][100] = { "open", "close", "up", "down", "facebook", "youtube", "news", "bookmark", "history", "read", "stop", "weather", "cricket" };
char* code_book_file = "codebook_32_size.txt";
char observation_sequence_file[100] = "observation_sequence_generated.txt";
char obs_sequence_file[100] = "observation_sequence_live.txt";

const int total_window = (FRAME_SIZE + (FRAMES - 1) * 80);
long double code_book[CODE_BOOK_SIZE][12] = { 0 };
long double c_prime_arr[T][12] = { 0 };
long double no_of_samples1 = 0;

int frame_cutting_index = 1;
int start_marker = 0;
int end_marker = 0;

char a_average_file[100] = "a_i_j_final.txt";
char b_average_file[100] = "b_i_j_final.txt";
char pi_average_file[100] = "pi_final.txt";

//Used Files
char input_file[100] = "recorded_digits/184101027_0_1";
char* reference_file[5] = { "ref_file_prime_a.txt", "ref_file_prime_e.txt", "ref_file_prime_i.txt", "ref_file_prime_o.txt", "ref_file_prime_u.txt" };
char normalized_file[100] = "Normalized_";
char silence_file[100] = "Seminar_silence.txt";
char trimmed_file[100] = "trim_";
char* ri_file = "ri_file.txt";
char* ai_file = "ai_file.txt";
char* ci_file = "ci_file.txt";
char* c_prime_file = "c_prime.txt";
char* universe_file = "universe_digits.txt";
char* hamming_file = "Hamming_window.txt";
char save_filename[100],iteration_file[100];

long double tokhura_weight[12] = { 1, 3, 7, 13, 19, 22, 25, 33, 42, 50, 56, 61 };
long double tokhura_dist[5] = { 0 };
const int p = 12;
int flag = 0;
int max_sample_index = 0;
int remove_header_counter = 5;
long double hamming_window[320] = { 0 };
long double current_value;
long double sum_samples = 0;
double dc_shift_value = 0;
double normalization_ratio = 1;
long int zcr_count = 0;
long int no_of_samples = 0;
long double max_sample_value = 1;
long double sample_array1[50000] = { 0 };
long double sample_array[total_window] = { 0 };
long double r[12] = { 0 }, k[12] = { 0 }, alpha[13][13] = { 0 }, E[13] = { 0 }, a[13] = { 0 }, c[12] = { 0 }, c_prime[12] = { 0 }, w[12] = { 0 };
int obs[T + 1];

long double a_i_j[N + 1][N + 1];
long double b_i_j[N + 1][M + 1];
long double pi[N + 1];

long double alpha_arr[N + 1][T + 1];

char* alpha_file = "alpha.txt";
long double Pobs_model = 0;
long double max_Pobs_model = 0;

int output_model_name = 0;

//TO READ CODEBOOK FROM FILE
void read_code_book(){
	in.open(code_book_file);
	for (int i = 0; i < CODE_BOOK_SIZE; i++){
		for (int j = 0; j < 12; j++){
			in >> code_book[i][j];
		}
	}
	in.close();
}

//TO PRINT CODEBOOK
void print_code_book(){
	//cout << "\n**************** Code Book **************" << endl;
	for (int i = 0; i < CODE_BOOK_SIZE; i++){
		for (int j = 0; j < 12; j++){
			cout << " " << code_book[i][j];
		}
		cout << endl;
	}
}

//To Remove header from the input files
void remove_header(ifstream& in){
	if (flag){
		remove_header_counter = 5;
		string temp = "";
		while (getline(in, temp) && remove_header_counter){
			remove_header_counter--;
		}
		flag = 0;
	}
	else{
		remove_header_counter = 5;
		string temp = "";
		while (getline(in, temp) && remove_header_counter){
			remove_header_counter--;
		}
	}
}

//To calculate DC Shift
void calculate_dc_shift(){
	int count = 0;
	sum_samples = 0;
	no_of_samples = 0;

	in.open(silence_file);
	flag = 1;
	//cout << "\n................Calculating DC shift..................." << endl;
	remove_header(in);
	string temp = "";

	while (in >> temp && !remove_header_counter){
		current_value = stod(temp);

		count++;
		if (count >= 200){
			sum_samples += current_value;
			no_of_samples++;
		}
	}

	//cout << "\nNo of samples:" << no_of_samples << endl;
	//cout << "Sum:" << sum_samples << endl;
	dc_shift_value = sum_samples / no_of_samples;
	//cout << "DC Shift value:" << dc_shift_value << endl;
	in.close();
}

//To calculate normalization ratio
void calculate_normalization_ratio(){
	int index_count = 0;
	max_sample_value = 0;
	max_sample_index = 0;
	no_of_samples1 = 0;
	//cout << "\n............Reading from " << input_file << "................" << endl;
	in.open(input_file);
	remove_header(in);
	string temp = "";
	//cout << "\n............Calculating Normalization Ratio................" << endl;
	while (in >> temp && !remove_header_counter){
		index_count++;
		no_of_samples1++;
		current_value = stold(temp);

		//Saving maximum index value and maximum sample value
		if (abs(current_value) > max_sample_value){
			max_sample_value = abs(current_value);
			max_sample_index = index_count;
		}
	}
	//cout << "Max Sample value:" << max_sample_value << endl;
	//cout << "Max Sample index:" << max_sample_index << endl;

	//Calculating normalization ratio
	normalization_ratio = 5000.0 / max_sample_value;
	//cout << "Normalization ratio:" << normalization_ratio << endl;
	in.close();
}

//To get Hamming Windows values to array from the pre-calculated File
void get_hamming_window(){
	int index_count = 0;
	in.open(hamming_file);
	string temp = "";
	while (in >> temp){
		hamming_window[index_count++] = stod(temp);
	}
	in.close();
}

//To remove DC Shift and normalize
void dc_normalize(){

	//Initializing Markers as + or - 640 from the max sample value
	//start_marker = max_sample_index - (total_window / 2);
	//end_marker = max_sample_index + (total_window / 2);

	/*if (start_marker < 0){
	cout << "....................................................................................Adjusting Beg........." << endl;
	start_marker = 1;
	end_marker = total_window + 1;
	}

	if (end_marker>no_of_samples1){
	cout << "....................................................................................Adjusting End........." << endl;
	end_marker = no_of_samples1 + 1;
	start_marker = no_of_samples1 - total_window;
	}

	cout << "Start Marker:" << start_marker << endl;
	cout << "End Marker:" << end_marker << endl;*/


	//cout << "Total Window:" << total_window << endl;

	int index_count = 0;
	int arr_index = 0;
	in.open(input_file);
	remove_header(in);

	string temp = "";

	//cout << "\n........Removing DC shift and Normalizing File.........." << endl;
	char index[3];
	//sprintf(index, "%d", utterence);
	strcpy(normalized_file, "normalized");
	//strcat(normalized_file, index);

	strcat(normalized_file, ".txt");
	//cout << normalized_file << endl;

	out.open(normalized_file);
	//out1.open("sample_array.txt");
	//Subtracting DC shift and Multiplying Normalization ratio
	while (in >> temp){

		index_count++;

		current_value = stold(temp);
		current_value = current_value - dc_shift_value;
		current_value = current_value * normalization_ratio;

		/*if (index_count >= start_marker && index_count < end_marker){
		sample_array[arr_index++] = current_value;
		out1 << sample_array[arr_index - 1] << endl;
		}*/

		//Writing the Normalized values to file "normalized.txt"
		sample_array1[index_count - 1] = current_value;
		out << to_string(current_value) << endl;
	}
	//cout << "First:"<<sample_array[0] << endl;
	//cout << "Last:"<<sample_array[arr_index - 1] << endl;
	//cout << "arr_index" << arr_index<<endl;
	//out1.close();
	out.close();

	in.close();
}
void trim_waveform1(){
	int i;
	long double total_energy;
	//interval is for windows
	//writing count is for final write after markers are found
	//intital shift pointer is for shifting by 1000 values
	//shift pointer count is for reading input file
	total_energy = 0;
	int interval = total_window, index_count = 0, arr_index = 0;
	long double total_max_energy = 0;
	int initial_shift_pointer = 0, shift_pointer_count = 0, writing_count = 0;
	long double arr_energy[50];
	long count1 = 0;
	string temp = "";
	//cout << "\n................Trimming Waveform..................." << endl;
	int count = interval;
	//in.open(normalized_file);
	out.open("energy_trim.txt");

	//window shifting logic
	//while (in >> temp){

	for (i = 0; i < no_of_samples1; i++){
		count1++;
		//if (shift_pointer_count >= initial_shift_pointer){

		//current_value = stold(temp);
		current_value = sample_array1[i];
		current_value *= 0.1;
		total_energy += (current_value*current_value);
		count--;
		if (!count){

			out << to_string(total_energy) << endl;

			if (total_energy > total_max_energy){
				total_max_energy = total_energy;
				start_marker = initial_shift_pointer;
				end_marker = initial_shift_pointer + interval;

			}

			total_energy = 0;
			count = interval;
			initial_shift_pointer += 100;
			i = initial_shift_pointer - 1;
			shift_pointer_count = 0;

			//in.clear();
			//in.seekg(0);
		}
		//}
		//shift_pointer_count++;
	}
	out.close();
	//in.close();

	in.open(normalized_file);
	//cout << "Start Marker : " << start_marker << endl;
	//cout << "End Marker : " << end_marker << endl;

	
	char index[3];
	//sprintf(index, "%d", utterence);
	strcpy(trimmed_file, "trim");
	//strcat(trimmed_file, index);
	strcat(trimmed_file, ".txt");
	//cout << "Trimmed:" << trimmed_file << endl;

	out1.open(trimmed_file);
	while (in >> temp){
		index_count++;
		current_value = stold(temp);
		//current_value = current_value - dc_shift_value;
		//current_value = current_value * normalization_ratio;

		if (index_count >= start_marker && index_count < end_marker){
			sample_array[arr_index++] = current_value;
			out1 << sample_array[arr_index - 1] << endl;
		}
	}
	out1.close();

	//Writing to new file from start to end marker
	/*out1.open(trimmed_file);
	while (in >> temp){
	writing_count++;
	if (writing_count >= start_marker && writing_count <= end_marker){
	cout << temp << endl;
	out1 << temp << endl;
	}
	else{
	continue;
	}
	}
	out1.close();*/
	in.close();
}
void trim_waveform2(){

	long double total_energy;
	//interval is for windows
	//writing count is for final write after markers are found
	//intital shift pointer is for shifting by 1000 values
	//shift pointer count is for reading input file
	total_energy = 0;
	int interval = total_window, index_count = 0, arr_index = 0;
	long double total_max_energy = 0;
	int initial_shift_pointer = 0, shift_pointer_count = 0, writing_count = 0;
	long double arr_energy[50];

	string temp = "";
	//cout << "\n................Trimming Waveform..................." << endl;
	int count = interval;

	in.open(normalized_file);
	out.open("energy_trim.txt");
	long double current_value1 = 0;
	//window shifting logic
	while (in >> temp){
		if (shift_pointer_count >= initial_shift_pointer){

			current_value = stold(temp);
			current_value *= 0.1;
			total_energy += (current_value*current_value);
			count--;
			if (!count){

				out << to_string(total_energy) << endl;

				if (total_energy > total_max_energy){
					total_max_energy = total_energy;
					start_marker = initial_shift_pointer;
					end_marker = initial_shift_pointer + interval;

				}

				total_energy = 0;
				count = interval;
				initial_shift_pointer += 100;
				shift_pointer_count = 0;
				in.clear();
				in.seekg(0);
			}
		}
		shift_pointer_count++;
	}
	out.close();
	in.clear();
	in.seekg(0);

	//cout << "Start Marker : " << start_marker << endl;
	//cout << "End Marker : " << end_marker << endl;

	char index[3];
	//sprintf(index, "%d", utterence);
	strcpy(trimmed_file, "trim");
	//strcat(trimmed_file, index);
	strcat(trimmed_file, ".txt");
	out1.open(trimmed_file);
	while (in >> temp){
		index_count++;
		current_value = stold(temp);
		//current_value = current_value - dc_shift_value;
		//current_value = current_value * normalization_ratio;

		if (index_count >= start_marker && index_count < end_marker){
			sample_array[arr_index++] = current_value;
			out1 << sample_array[arr_index - 1] << endl;
		}
	}
	out1.close();

	//Writing to new file from start to end marker
	/*out1.open(trimmed_file);
	while (in >> temp){
	writing_count++;
	if (writing_count >= start_marker && writing_count <= end_marker){
	cout << temp << endl;
	out1 << temp << endl;
	}
	else{
	continue;
	}
	}
	out1.close();*/
	in.close();
}

//Calculating Ri's
void calculate_Ris(){
	//cout << "\n........Writing Ri's to file.........." << endl;
	int count = 0;
	long double first_value = 0;
	long double second_value = 0;
	string temp;

	out.open(ri_file);

	//Calculating Ri and also multiplying with hamming window appropriately
	for (int k = 0; k < FRAMES; k++){
		for (int j = 0; j <= p; j++){
			r[j] = 0;
			for (int i = 0; i < FRAME_SIZE - j; i++){
				first_value = sample_array[i + (80 * k)];
				second_value = sample_array[(i + j) + (80 * k)];
				//cout << "First:" << first_value << endl;
				//cout << "Second:" << second_value << endl;
				
				r[j] = r[j] + ((first_value*hamming_window[i])*(second_value*hamming_window[i + j]));
			}
			//cout << r[j] << endl;
		}
		for (int j = 0; j <= p; j++){
			//printf("R[%d] : %Lf \n", j, r[j]);
			out << fixed << r[j] << " ";
			r[j] = 0;
		}
		out << endl;
		//cout << endl;
	}
	out.close();
	cout << endl;
}

//Calculate Ai's using Levenson-Durbin
void levenson_durbin(long double r[]){
	int i, j;
	long double summation = 0;

	E[0] = r[0];
	for (i = 1; i <= p; i++){
		summation = 0.0;
		for (j = 1; j <= i - 1; j++){
			summation += alpha[j][i - 1] * r[i - j];
		}
		k[i] = (r[i] - summation) / E[i - 1];
		alpha[i][i] = k[i];
		for (j = 1; j <= i - 1; j++){
			alpha[j][i] = alpha[j][i - 1] - (k[i] * alpha[i - j][i - 1]);
		}
		E[i] = (1 - (k[i] * k[i]))*E[i - 1];
	}
	//cout << endl;

	//Calculating the Ai's as the last last column of index i.e. 12th column of matrix alpha
	out.open(ai_file, std::ios_base::app);
	a[0] = 0.0;
	out << a[0] << " ";
	for (int i = 1; i <= p; i++){
		a[i] = alpha[i][12];
		//printf("\nA[%d] : %Lf ", i, a[i]);
		out << fixed << a[i] << " ";
	}
	out << endl;
	out.close();
}

//calling levenson Durbin for 85 frames one after the another
void calculate_Ais(){
	//cout << "\n........Writing Ai's Frames to file.........." << endl;
	int abc = 1;
	string temp;
	int count = 0;
	in.open(ri_file);
	out.open(ai_file);
	out.close();
	for (int k = 0; k < FRAMES; k++){
		while (count <= 12){
			in >> temp;
			r[count] = stod(temp);
			count++;
		}
		levenson_durbin(r);
		count = 0;
	}
	in.close();
}

//Calculate Ci's
void calculate_Cis(){
	//cout << "\n........Writing Ci's for 5 Frames to files.........." << endl;
	string temp;
	long double summation = 0;
	int count = 0;
	long double a1 = 0.0, a2 = 0.0;
	in.open(ai_file);
	in1.open(ri_file);
	out.open(ci_file);
	out.close();

	//Calculating Cis by reading files of Ais and Ris
	out.open(ci_file/*, std::ios_base::app*/);
	for (int k = 0; k < FRAMES; k++){
		while (count <= 12){
			in >> temp;
			//cout << temp << endl;
			a[count] = stod(temp);
			in1 >> temp;
			r[count] = stod(temp);
			count++;
		}
		c[0] = 2 * log(r[0]) / log(2.0);
		for (int m = 1; m <= p; m++){
			summation = 0;
			for (int k = 1; k <= m - 1; k++)
			{
				summation += ((double)k / m)*c[k] * a[m - k];
			}
			c[m] = a[m] + summation;
		}
		for (int m = 1; m <= p; m++){
			out << fixed << c[m] << " ";
			//printf("\nC[%d] : %Lf ", m, c[m]);
			c[m] = 0;
		}
		//cout << endl;
		out << endl;
		count = 0;
	}
	out.close();
	in1.close();
	in.close();
}

//To calculate C Prime(Raised Sine wave)
void calculate_c_prime(){
	//cout << "\n........Writing Ci Prime Frames to file.........." << endl;
	string temp;
	int count = 0;
	in.open(ci_file);
	//out.open(c_prime_file);
	//out.close();

	//out1.open(universe_file, std::ios_base::app);

	//Calculting CiPrimes by reading Ci values from File
	out1.open(c_prime_file);
	out.open(universe_file, std::ios_base::app);
	for (int k = 0; k < FRAMES; k++){
		while (count < 12){
			in >> temp;
			c[count] = stod(temp);
			count++;
		}
		for (int m = 1; m <= p; m++){
			w[m] = 1.0 + 6.0 * sin((22.0 / 7.0)*m / 12.0);
			c_prime[m] = c[m - 1] * w[m];
		}
		for (int m = 1; m <= p; m++){
			//printf("\nC_Prime[%d] : %Lf ", m, c_prime[m]);
			out << c_prime[m] << " ";
			out1 << c_prime[m] << " ";
			//out1 << c_prime[m] << " ";
			c[m] = 0;
			c_prime[m] = 0;
		}
		//cout << endl;
		out << endl;
		out1 << endl;
		//out1 << endl;
		count = 0;
	}
	in.close();
	out.close();
	out1.close();

	//out1.close();
}

//Calculating Tokhura's Distance Using Code Book
void calculate_tokhura_distance(long double c[12], int index){
	int  min_index = 0;
	long double min = 99999;
	long double sum[CODE_BOOK_SIZE] = { 0 };
	string temp, temp1;

	for (int j = 0; j < CODE_BOOK_SIZE; j++){
		for (int i = 0; i < 12; i++){
			sum[j] += tokhura_weight[i] * (c[i] - code_book[j][i])*(c[i] - code_book[j][i]);
		}
		if (sum[j] < min){
			min = sum[j];
			min_index = j;
		}
	}

	obs[index] = min_index + 1;

	cout << obs[index] << " ";
}

void read_ci_values(){
	int i, j;
	string temp;
	in.open(c_prime_file);
	for (i = 0; i < T; i++){
		for (j = 0; j < 12; j++){
			in >> temp;
			c_prime_arr[i][j] = stold(temp);
		}
	}
	in.close();
}

void write_observation_sequence(){

	int i;
	for (i = 0; i < T; i++)
		out << obs[i] << "\t";
	out << endl;
}

void generate_observation_sequence(){
	read_ci_values();
	int i;
	cout << "\nObservation Sequence:" << endl;
	for (i = 0; i < FRAMES; i++){
		calculate_tokhura_distance(c_prime_arr[i], i);
	}
	//write_observation_sequence();
}

void calculate_cepstral_values(){
	get_hamming_window();
	calculate_dc_shift();
	calculate_normalization_ratio();
	dc_normalize();
	trim_waveform1();
	calculate_Ris();
	calculate_Ais();
	calculate_Cis();
	calculate_c_prime();
}

void make_universe(){
	int i, j;

	//for (i = 0; i <=9; i++){
	char index[3];
	strcpy(obs_sequence_file, "observation_sequence_live");
	//sprintf(index, "%d", i);
	//strcat(obs_sequence_file, index);
	strcat(obs_sequence_file, ".txt");

	for (j = 1; j <= 30; j++){

		strcpy(input_file, "Input/youtube/youtube_");
		/*sprintf(index, "%d", i);
		strcat(input_file, index);
		strcat(input_file, "_");*/
		sprintf(index, "%d", j);
		strcat(input_file, index);
		strcat(input_file, ".txt");
		cout << "\n:::::INPUT:" << obs_sequence_file << endl;

		calculate_cepstral_values();
		read_code_book();
		generate_observation_sequence();
		out.open(obs_sequence_file, std::ios_base::app);
		write_observation_sequence();
		out.close();
	}

	//}
}


//TO READ A MATRIX FROM FILE
void read_Aij_values(char filename[]){
	in.open(filename);
	string temp;
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			in >> temp;
			a_i_j[i][j] = stold(temp);
		}
	}
	/*cout << "A:" << endl;
	for (int i = 1; i <= N; i++){
	for (int j = 1; j <= N; j++){
	printf("%.30Lf  ", a_i_j[i][j]);
	}
	cout << endl;
	}*/
	in.close();
}

//TO READ B MATRIX FROM FILE
void read_Bij_values(char filename[]){
	in.open(filename);
	string temp;
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			in >> temp;
			b_i_j[i][j] = stold(temp);
		}
	}
	/*cout << "B:" << endl;
	for (int i = 1; i <= N; i++){
	for (int j = 1; j <= M; j++){
	printf("%Le ", b_i_j[i][j]);
	}
	cout << endl;
	}*/
	in.close();
}

//TO READ PI VALUES FROM FILE
void read_pi_values(char filename[]){
	in.open(filename);
	string temp;
	int index = 1;
	while (in >> temp){
		pi[index++] = stold(temp);
	}
	in.close();
	/*cout << "Pi values:" << endl;
	for (int i = 1; i < index; i++)
	printf("%.30Lf\n", pi[i]);*/
}


void read_average_model(int iteration){
	char index[3];
	strcpy(a_average_file, "Input/average_models/A_avg_");
	//sprintf(index, "%d", iteration);
	strcat(a_average_file, words[iteration]);
	strcat(a_average_file, ".txt");
	//cout << a_average_file << endl;
	read_Aij_values(a_average_file);

	strcpy(b_average_file, "Input/average_models/B_avg_");
	//sprintf(index, "%d", iteration);
	strcat(b_average_file, words[iteration]);
	strcat(b_average_file, ".txt");
	//cout << b_average_file << endl;
	read_Bij_values(b_average_file);

	strcpy(pi_average_file, "Input/average_models/pi_avg_");
	//sprintf(index, "%d", iteration);
	strcat(pi_average_file, words[iteration]);
	strcat(pi_average_file, ".txt");
	//cout << pi_average_file << endl;
	read_pi_values(pi_average_file);
}

//TO PERFORM THE FORWARD PROCEDURE
void forward_procedure(int iteration){


	int i, j, t;
	long double sum = 0;
	//for (i = 0; i < T; i++)
	//cout << "OBSss:" << obs[i] << " ";

	/*for (i = 1; i <= T;i++)
	for (j = 1; j <= N; j++)
	alpha_arr[i][j] = 0;*/

	for (i = 1; i <= N; i++){
		alpha_arr[i][1] = pi[i] * b_i_j[i][obs[0]];
		//cout << "\n" << b_i_j[i][obs[1]] << endl;
		//cout << pi[i] << endl;
		//cout << alpha[i][1] << endl;
	}

	//CALCULATING ALPHA MATRIX
	for (t = 1; t <= T - 1; t++){
		for (j = 1; j <= N; j++){
			sum = 0;
			for (i = 1; i <= N; i++){

				sum += alpha_arr[i][t] * a_i_j[i][j];
			}

			alpha_arr[j][t + 1] = sum * b_i_j[j][obs[t]];
		}
	}

	//PRINTING ALPHA
	/*cout << "\nAlpha Matrix:" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= T; j++){
	printf("%Le ", alpha_arr[i][j]);
	}
	cout << endl;
	}*/



	//WRITING ALPHA TO FILE
	out.open(alpha_file);
	for (i = 1; i <= T; i++){
		for (j = 1; j <= N; j++){
			out << /*fixed << setprecision(FILE_DECIMAL_PRECISION_VALUE) <<*/ alpha_arr[j][i] << " ";
		}
		out << endl;
	}
	out.close();


	//CALCULATING PROBABILITY OF OBSERVATION SEQUENCE GIVEN MODEL
	sum = 0;
	for (i = 1; i <= N; i++){
		sum += alpha_arr[i][T];
	}



	Pobs_model = sum;
	if (Pobs_model > max_Pobs_model){
		max_Pobs_model = Pobs_model;
		output_model_name = iteration;
	}
	cout << "\nDigit:" << iteration << "\tP(obs/model) : " << Pobs_model;
}


void solution_to_problem1(int iteration){
	forward_procedure(iteration);
}


//################################################    TRAINING FUNCTIONS #############################################################

#define B_I_J_THRESHOLD pow(10.0,-30.0)

char a_file_final[100] = "a_i_j_final.txt";
char b_file_final[100] = "b_i_j_final.txt";
char pi_file_final[100] = "pi_final.txt";

char* a_file = "a_i_j.txt";
char* b_file = "b_i_j.txt";
char* pi_file = "pi.txt";


char* beta_file = "beta.txt";
char* qstar1_file = "state_sequence_viterbi.txt";

long double beta[N + 1][T + 1];

long double delta[N + 1][T + 1];
long double pstar = 0;
long double previous_pstar = 0;
long double pstar_difference = 0;
int psi[N + 1][T + 1];
int qstar, qstar1[T + 1];

long double Xi[N + 1][N + 1][T];
long double gamma[N + 1][T + 1];
long double gamma1[N + 1][T + 1];

long double a_i_j_average[N + 1][N + 1];
long double b_i_j_average[N + 1][M + 1];
long double pi_average[N + 1];


//TO APPLY THRESHOLD VALUE TO MAKE VALUES NON-ZERO
void apply_threshold_to_Bij(){
	int i, j;
	long double diff;
	long double max;
	int max_i=0;
	int max_j=0;
	for (i = 1; i <= N; i++){
		diff = 0;
		max = 0;
		for (j = 1; j <= M; j++){
			if (b_i_j[i][j] > max){
				max = b_i_j[i][j];
				max_i = i;
				max_j = j;
			}
			if (b_i_j[i][j] < B_I_J_THRESHOLD){
				diff += b_i_j[i][j] - B_I_J_THRESHOLD;
				b_i_j[i][j] = B_I_J_THRESHOLD;
			}
		}
		b_i_j[max_i][max_j] = max;
	}
}

//TO PRINT MATRIX OF DIMENSION N*T
void print_matrix(long double matrix[N + 1][T + 1]){
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= T; j++){
			cout << "\n" << matrix[i][j];
			//printf("%.30Lf\n", matrix[i][j]);
		}
		cout << endl << endl;
	}
}

//TO WRITE THE UPDATED A VALUES
void write_final_A_values(char filename[]){
	out.open(filename);
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){
			out << a_i_j[i][j] << " ";
		}
		out << endl;
	}
	out.close();
}

//TO WRITE THE UPDATED B VALUES
void write_final_B_values(char filename[]){
	out.open(filename);
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			out << b_i_j[i][j] << " ";
		}
		out << endl;
	}
	out.close();
}

//TO WRITE THE UPDATED PI VALUES
void write_final_pi_values(char filename[]){
	out.open(filename);
	int i, j;
	for (i = 1; i <= N; i++){
		out << pi[i] << " ";
	}
	out.close();
}

//DUMP FINAL MODAL VALUES
void dump_final_model(int iteration, int iteration1){
	char index[3];

	strcpy(a_file_final, "output/models_");
	//sprintf(index, "%d", iteration1);
	strcat(a_file_final, words[iteration1]);
	strcat(a_file_final, "/A_");
	sprintf(index, "%d", iteration);
	strcat(a_file_final, index);
	strcat(a_file_final, ".txt");
	cout << a_file_final << endl;
	write_final_A_values(a_file_final);

	strcpy(b_file_final, "output/models_");
	//sprintf(index, "%d", iteration1);
	strcat(b_file_final, words[iteration1]);
	strcat(b_file_final, "/B_");
	sprintf(index, "%d", iteration);
	strcat(b_file_final, index);
	strcat(b_file_final, ".txt");
	//cout << b_file_final << endl;
	write_final_B_values(b_file_final);

	strcpy(pi_file_final, "output/models_");
	//sprintf(index, "%d", iteration1);
	strcat(pi_file_final, words[iteration1]);
	strcat(pi_file_final, "/pi_");
	sprintf(index, "%d", iteration);
	strcat(pi_file_final, index);
	strcat(pi_file_final, ".txt");
	//cout << pi_file_final << endl;
	write_final_pi_values(pi_file_final);
}

//TO READ A MATRIX FROM FILE
/*void read_Aij_values(char filename[]){
	in.open(filename);
	string temp;
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			in >> temp;
			a_i_j[i][j] = stold(temp);
		}
	}
	/*cout << "A:" << endl;
	for (int i = 1; i <= N; i++){
	for (int j = 1; j <= N; j++){
	printf("%.30Lf  ", a_i_j[i][j]);
	}
	cout << endl;
	}
	in.close();
}*/

//TO READ B MATRIX FROM FILE
/*void read_Bij_values(char filename[]){
	in.open(filename);
	string temp;
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			in >> temp;
			b_i_j[i][j] = stold(temp);
		}
	}
	/*cout << "B:" << endl;
	for (int i = 1; i <= N; i++){
	for (int j = 1; j <= M; j++){
	printf("%.30Lf\n", b_i_j[i][j]);
	}
	cout << endl << endl;
	}
	in.close();
}*/

//TO READ PI VALUES FROM FILE
/*void read_pi_values(char filename[]){
	in.open(filename);
	string temp;
	int index = 1;
	while (in >> temp){
		pi[index++] = stold(temp);
	}
	in.close();
	/*cout << "Pi values:" << endl;
	for (int i = 1; i <= index; i++)
	printf("%.30Lf\n", pi[i]);
}*/


//TO READ OBSERVATION SEQUENCE FROM FILE
void read_observation_values(ifstream &inn){

	string temp;
	for (int i = 1; i <= T; i++){
		inn >> temp;
		obs[i] = stoi(temp);
		
	}
	/*cout << "\nObservation Sequence:" << endl;
	for (int i = 1; i <= T; i++){
	cout << obs[i] << " ";
	}
	cout << endl;*/

}


//TO READ THE INTIAL MODEL
void read_initial_model(){
	read_pi_values(pi_file);
	read_Aij_values(a_file);
	read_Bij_values(b_file);
}

//TO PERFORM THE FORWARD PROCEDURE
void forward_procedure(){
	int i, j, t;
	long double sum = 0;
	for (i = 1; i <= N; i++)
		alpha[i][1] = pi[i] * b_i_j[i][obs[1]];

	//CALCULATING ALPHA MATRIX
	for (t = 1; t <= T - 1; t++){
		for (j = 1; j <= N; j++){
			sum = 0;
			for (i = 1; i <= N; i++){
				sum += alpha[i][t] * a_i_j[i][j];
			}
			alpha[j][t + 1] = sum * b_i_j[j][obs[t + 1]];
		}
	}

	//PRINTING ALPHA
	//cout << "\nAlpha Matrix:" << endl;
	/*for (i = 1; i <= N; i++){
	for (j = 1; j <= T; j++){
	printf("%Le\n", alpha[i][j]);
	}
	cout << endl << endl;
	}*/

	//WRITING ALPHA TO FILE
	out.open(alpha_file);
	for (i = 1; i <= T; i++){
		for (j = 1; j <= N; j++){
			out << /*fixed << setprecision(FILE_DECIMAL_PRECISION_VALUE) <<*/ alpha[j][i] << " ";
		}
		out << endl;
	}
	out.close();


	//CALCULATING PROBABILITY OF OBSERVATION SEQUENCE GIVEN MODEL
	sum = 0;
	for (i = 1; i <= N; i++){
		sum += alpha[i][T];
	}

	Pobs_model = sum;
	//cout << "\nP(obs/model) : " << Pobs_model <<endl;
}

//TO PERFORM THE BACKWARD PROCEDURE
void backward_procedure(){
	int i, j, t;
	long double sum = 0;
	for (i = 1; i <= N; i++)
		beta[i][T] = 1.0;

	//CALCULATING BETA MATRIX
	for (t = T - 1; t >= 1; t--){
		for (i = 1; i <= N; i++){
			beta[i][t] = 0;
			for (j = 1; j <= N; j++){
				beta[i][t] += a_i_j[i][j] * b_i_j[j][obs[t + 1]] * beta[j][t + 1];
			}
		}
	}

	//PRINTING BETA MATRIX
	/*cout << "\Beta Matrix:" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= T; j++){
	printf("%.30Lf\n", beta[i][j]);
	}
	cout << endl << endl;
	}*/

	//WRITING BETA TO FILE
	out.open(beta_file);
	for (i = 1; i <= T; i++){
		for (j = 1; j <= N; j++){
			out << /*fixed << setprecision(FILE_DECIMAL_PRECISION_VALUE) <<*/ beta[j][i] << " ";
		}
		out << endl;
	}
	out.close();
}

//PERFORMS VIETERBI ALGORITHM
void perform_viterbi(){
	int i, j, t;
	long double max = 0, temp = 0;

	//STEP 1 : INTITIALIZATION
	for (i = 1; i <= N; i++){
		delta[i][1] = pi[i] * b_i_j[i][obs[1]];
		psi[i][1] = 0;
	}

	//STEP 2 : RECURSION 
	for (t = 2; t <= T; t++){
		for (j = 1; j <= N; j++){
			max = 0;
			for (i = 1; i <= N; i++){
				temp = delta[i][t - 1] * a_i_j[i][j];
				if (temp > max){
					max = temp;
					psi[j][t] = i;
				}
			}
			delta[j][t] = max*b_i_j[j][obs[t]];
		}
	}

	//PRINTING DELTA VALUES
	//cout << "\nDelta" << endl;
	//print_matrix(delta);


	//PRITING PSI VALUES
	/*cout << "\nPsi" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= T; j++){
	cout << " " << psi[i][j];
	}
	cout << endl;
	}*/

	//STEP 3 : TERMINATION
	max = 0;
	for (i = 1; i <= N; i++){
		if (delta[i][T] > max){
			max = delta[i][T];
			pstar = delta[i][T];
			qstar = i;
		}
	}


	//cout << "\nQ Star[T]:" << qstar << endl << endl;
	qstar1[T] = qstar;
	for (t = T - 1; t >= 1; t--){
		qstar1[t] = psi[qstar1[t + 1]][t + 1];
	}

	//STEP 4 : PATH
	/*cout << "\nState Seqeuence (Viterbi) : " << endl;
	for (t = 1; t <= T; t++){
	cout << " " << qstar1[t];
	}*/
	//cout << "\nP Star : " << pstar << endl;
	//cout << "\nP Star (Previous) : " << previous_pstar << endl;


	pstar_difference = pstar - previous_pstar;
	//cout << "\nDifference : " << pstar_difference << endl;
	previous_pstar = pstar;


	//WRITING SEQUENCE TO FILE
	out.open(qstar1_file);
	for (t = 1; t <= T; t++){
		out << qstar1[t] << endl;
	}
	out.close();
}

//TO CALCULATE XI VALUES
void calculate_xi_values(){
	int i, j, t;
	long double summation[T];

	//CALCULATING SUMMATION ARRAY
	for (t = 1; t <= T - 1; t++){
		summation[t] = 0;
		for (i = 1; i <= N; i++){
			for (j = 1; j <= N; j++){
				summation[t] += alpha[i][t] * a_i_j[i][j] * b_i_j[j][obs[t + 1]] * beta[j][t + 1];

			}
		}
	}

	for (t = 1; t <= T - 1; t++){
		//cout << "Summation:" << summation[t] << endl;
		//printf("%Le\n", summation[t]);
	}


	//CALCULATING XI MATRIX
	for (t = 1; t <= T - 1; t++){
		for (i = 1; i <= N; i++){
			for (j = 1; j <= N; j++){

				Xi[i][j][t] = (alpha[i][t] * a_i_j[i][j] * b_i_j[j][obs[t + 1]] * beta[j][t + 1]) / summation[t];
			}
		}
	}

	//PRINTING XI MATRIX
	/*cout << "\nXi Matrix : " << endl;
	for (t = 1; t <= T-1; t++){
	for (i = 1; i <= N; i++){
	for (j = 1; j <= N; j++){
	printf("%Le ", Xi[i][j][t]);
	//cout << Xi[i][j][t] << " ";
	}
	cout << endl;
	}
	cout << endl << endl;
	}*/
}

//TO CALULATE PROBABILITY OF ENDING AT STATE I : GAMMA MATRIX
void calculate_gamma_values(){
	int i, j, t;

	//CALCULATING GAMMA MATRIX
	for (t = 1; t <= T - 1; t++){
		for (i = 1; i <= N; i++){
			gamma[i][t] = 0;
			for (j = 1; j <= N; j++){
				gamma[i][t] += Xi[i][j][t];
			}
		}
	}

	//PRINTING GAMMA MATRIX
	/*cout << "\nGamma Matrix : " << endl;
	for (t = 1; t <= T - 1; t++){
	for (i = 1; i <= N; i++){
	cout << gamma[i][t] << " ";
	}
	cout << endl;
	}*/
}

//TO CALCULATE GAMMA MATRIX (BASED ON ALPHA-BETA)
void calculate_gamma(){
	int i, j, k, t;
	long double summation = 0;
	//CALCULATING GAMMA MATRIX
	for (i = 1; i <= T; i++){
		for (j = 1; j <= N; j++){
			summation = 0;
			for (k = 1; k <= N; k++){
				summation += alpha[k][i] * beta[k][i];
			}
			gamma[j][i] = (alpha[j][i] * beta[j][i]) / summation;
		}

	}
	/*cout << "\nGamma Matrix : " << endl;
	for (t = 1; t <= T - 1; t++){
	for (i = 1; i <= N; i++){
	cout << gamma[i][t] << " ";
	}
	cout << endl;
	}*/
}

//TO UPDATE MODEL PARAMTERS
void update_model_parameters(){
	int i, j, t;
	long double summation = 0;
	long double summation1 = 0;

	//UPDATING THE PI VALUES
	for (i = 1; i <= N; i++){
		pi[i] = gamma[i][1];
	}

	//PRINTING NEW PI VALUES
	/*cout << "\nNew Pi Values:" << endl;
	for (i = 1; i <= N; i++){
	cout <<" "<< pi[i];
	}*/

	//UPDATING THE A MATRIX VALUES
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){
			summation = 0;
			summation1 = 0;
			for (t = 1; t <= T - 1; t++){
				summation += Xi[i][j][t];
				summation1 += gamma[i][t];
			}
			a_i_j[i][j] = summation / summation1;
		}

	}

	//PRINTING THE A MATRIX VALUES
	/*cout << "\nNew A:" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= N; j++){
	cout << a_i_j[i][j] << " ";
	}
	cout << endl;
	}*/

	//UPDATING THE B MATRIX VALUES
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			summation = 0;
			summation1 = 0;
			for (t = 1; t <= T; t++){
				if (obs[t] == j)
					summation1 += gamma[i][t];
			}
			for (t = 1; t <= T; t++){
				summation += gamma[i][t];
			}
			b_i_j[i][j] = summation1 / summation;
		}
	}

	//APPLYING THRESHOLD FUNCTION TO UPDATE B MATRIX

	/*cout << "\n::New B:" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= M; j++){
	printf("%Le ", b_i_j[i][j]);
	//cout << b_i_j[i][j] << " ";
	}
	cout << endl;
	}*/

	apply_threshold_to_Bij();

	//PRINTING THE FINAL B MATRIX
	/*cout << "\n::New B:" << endl;
	for (i = 1; i <= N; i++){
	for (j = 1; j <= M; j++){
	printf("%Le ", b_i_j[i][j]);
	//cout << b_i_j[i][j] << " ";
	}
	cout << endl;
	}*/

}

//TO PERFORM EM PROCEDURE
void expectation_maximization(){
	calculate_xi_values();
	calculate_gamma();
	//calculate_gamma_values();
	update_model_parameters();
}

//SOLUTION TO PROBLEM 1 USING FORWARD AND BACKWARD PROCEDURES
void solution_to_problem1_1(){
	backward_procedure();
	forward_procedure();
}

//SOLUTION TO PROBLEM 2 USING VITERBI ALGORITHM
void solution_to_problem2(){
	perform_viterbi();

}

//SOLUTION TO PROBLEM 3 USING EM PROCEDURE
void solution_to_problem3(){
	expectation_maximization();
}

void add_values_to_average_model(){
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){

			a_i_j_average[i][j] += a_i_j[i][j];
		}
	}
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			b_i_j_average[i][j] += b_i_j[i][j];
		}
	}
	for (i = 1; i <= N; i++){
		pi_average[i] += pi[i];
	}
}

void average_of_average_model(int train_iterations){
	
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){
			a_i_j_average[i][j] /= train_iterations;

		}
	}
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			b_i_j_average[i][j] /= train_iterations;
		}
	}
	for (i = 1; i <= N; i++){
		pi_average[i] /= train_iterations;
	}
}

void clear_average_model_array(){
	int i, j;
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){
			a_i_j_average[i][j] = 0;
		}
	}
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			b_i_j_average[i][j] = 0;
		}
	}
	for (i = 1; i <= N; i++){
		pi_average[i] = 0;
	}
}

void dump_average_model(int iteration){
	int i, j;
	char index[3];

	strcpy(a_average_file, "output/average_models/A_avg_");
	//sprintf(index, "%d", iteration);
	strcat(a_average_file, words[iteration]);
	strcat(a_average_file, ".txt");
	//cout << a_average_file << endl;
	out.open(a_average_file);
	for (i = 1; i <= N; i++){
		for (j = 1; j <= N; j++){
			out << a_i_j_average[i][j] << " ";
		}
		out << endl;
	}
	out.close();

	strcpy(b_average_file, "output/average_models/B_avg_");
	//sprintf(index, "%d", iteration);
	strcat(b_average_file, words[iteration]);
	strcat(b_average_file, ".txt");
	//cout << b_average_file << endl;
	out.open(b_average_file);
	for (i = 1; i <= N; i++){
		for (j = 1; j <= M; j++){
			out << b_i_j_average[i][j] << " ";
		}
		out << endl;
	}
	out.close();

	strcpy(pi_average_file, "output/average_models/pi_avg_");
	//sprintf(index, "%d", iteration);
	strcat(pi_average_file, words[iteration]);
	strcat(pi_average_file, ".txt");
	//cout << pi_average_file << endl;
	out.open(pi_average_file);
	for (i = 1; i <= N; i++){
		out << pi_average[i] << " ";
	}
	out.close();
}

/*void read_average_model(int iteration){
	char index[3];
	strcpy(a_average_file, "output/average_models/A_avg_");
	sprintf(index, "%d", iteration);
	strcat(a_average_file, index);
	strcat(a_average_file, ".txt");
	//cout << a_average_file << endl;
	read_Aij_values(a_average_file);

	strcpy(b_average_file, "output/average_models/B_avg_");
	sprintf(index, "%d", iteration);
	strcat(b_average_file, index);
	strcat(b_average_file, ".txt");
	//cout << b_average_file << endl;
	read_Bij_values(b_average_file);

	strcpy(pi_average_file, "output/average_models/pi_avg_");
	sprintf(index, "%d", iteration);
	strcat(pi_average_file, index);
	strcat(pi_average_file, ".txt");
	//cout << pi_average_file << endl;
	read_pi_values(pi_average_file);
}*/


//###################################################### FINAL CALLING FUNCTIONS #########################################################

void initialize_iteration_files(int value){
	for (int i = 0; i < WORDS; i++){
		strcpy(iteration_file, "saved_recordings/iteration_");
		strcat(iteration_file, words[i]);
		strcat(iteration_file, ".txt");
		out.open(iteration_file);
		out << value;
		out.close();
	}
}

void save_current_recording(int iteration){
	int current_iteration;
	
	strcpy(iteration_file, "saved_recordings/iteration_");
	strcat(iteration_file, words[iteration]);
	strcat(iteration_file, ".txt");
	string temp;
	
	in.open(iteration_file);
	in >> temp;
	current_iteration = stoi(temp);
	

	in.close();

	current_iteration++;
	cout << "\nCurrent Iteration:"<<current_iteration << endl;
	system("cls");

	
	char index[3];
	strcpy(save_filename, "saved_recordings/");
	strcat(save_filename,words[iteration]);
	strcat(save_filename,"/");
	
	strcat(save_filename,words[iteration]);
	sprintf(index, "%d", current_iteration);
	strcat(save_filename,index);	
	strcat(save_filename, ".txt");
	cout << "File Saved At Location :"<<save_filename;


	in.open("input.txt");
	out.open(save_filename);
	while (in >> temp){
		//cout << temp;
		out << temp << endl;
	}
	out.close();
	in.close();

	int choice=2;
	cout << "\n\nDo you want to keep the current recording ?" << endl;
	cout << "Press 1. to Discard Else to Proceed" << endl;
	cin >> choice;
	if (choice != 1){
		out.open(iteration_file);
		out << current_iteration;
		out.close();
	}


}

void do_training(int initial_iteration){
	int current_iteration;
	int current_iteration_arr[WORDS];
	for (int i = 0; i < WORDS; i++){
		strcpy(iteration_file, "saved_recordings/iteration_");
		strcat(iteration_file, words[i]);
		strcat(iteration_file, ".txt");
		
		string temp;

		in.open(iteration_file);
		in >> temp;
		
		current_iteration_arr[i] = stoi(temp);				
		in.close();
	}

	int i, j, k, l;
	for (l = 0; l < 1; l++){
		cout << "\n....... Iteration No : " << l << " ........" << endl;
		for (k = 0; k < WORDS; k++){
			cout << "Calculating Average Model for : " << words[k] << endl;
			cout << "Training for " << (current_iteration_arr[k]-initial_iteration) << " Iterations" << endl;
			char index[3];

			strcpy(obs_sequence_file, "observation_sequences/");
			strcat(obs_sequence_file, words[k]);
			
			strcat(obs_sequence_file, ".txt");
			cout << obs_sequence_file << endl;

			in_obs.open(obs_sequence_file);

			for (j = initial_iteration+1; j <= current_iteration_arr[k]; j++){
				read_observation_values(in_obs);

				if (l == 0)
					read_initial_model();
				else
					read_average_model(k);

				for (i = 1; i <= 30; i++){
					/*cout << "\nObservation Sequence:" << endl;
					for (int i = 1; i <= T; i++){
					cout << obs[i] << " ";
					}*/
					
					solution_to_problem1_1();
					solution_to_problem2();
					solution_to_problem3();
				}

				add_values_to_average_model();
				
				dump_final_model(j, k);
				
			}
			average_of_average_model(current_iteration_arr[k]-initial_iteration);
			
			dump_average_model(k);
			clear_average_model_array();

			in_obs.close();
		}
	}
}

void do_recording(int index){
	int word_index = 3;
	int k;
	//make_universe();
	//strcpy(input_file, "Input/youtube/youtube_1.txt");
	cout << "\n.................................................................................................." << endl;
	//cout << "Reading Input from : " << input_file << endl;

	cout << "....... Recording will be for 3 seconds ......" << endl;
	//cout << "....... Recording Silence......" << endl;
	//system("Recording_Module.exe 3 silence.wav silence_file.txt");
	//cout << "\nSilence recorded. **Press Enter to record your DIGIT**" << endl;
	system("Recording_Module.exe 3 input.wav input.txt");
	cout << "\nRecording successfull **Press ENTER to proceed with program**" << endl;
	strcpy(input_file, "input.txt");
	strcpy(silence_file, "silence_file.txt");


	calculate_cepstral_values();
	read_code_book();
	//print_code_book();
	generate_observation_sequence();

	strcpy(obs_sequence_file, "observation_sequences/");
	strcat(obs_sequence_file,words[index]);
	strcat(obs_sequence_file, ".txt");

	out.open(obs_sequence_file,  std::ios_base::app);
	write_observation_sequence();
	out.close();
	/*output_model_name = 0;
	max_Pobs_model = 0;
	for (k = 0; k < WORDS; k++){
		read_average_model(k);
		solution_to_problem1(k);
	}

	cout << "\nDetected Digit:" << output_model_name << endl;*/
	//print_code_book();
}

int _tmain(int argc, _TCHAR* argv[])
{
	cout << "....... Recording Silence......" << endl;
	system("Recording_Module.exe 3 silence.wav silence_file.txt");

	while (1){
		int choice = 0, choice1 = 0;
		int default_iteration_value = 40;
		system("cls");
		cout << "\n\n******************** SELECT A CHOICE **********************" << endl;
		cout << "1.Record Words" << endl;
		cout << "2.Train Words" << endl;
		//cout << "3.Initialize Iteration Files" << endl;
		cout << "Enter a choice:" << endl;
		cin >> choice;

		system("cls");

		switch (choice)
		{
		case 1:

			cout << "\n********** SELECT A WORD YOU WISH TO SPEAK ************" << endl;
			cout << "1.Open" << endl;
			cout << "2.Close" << endl;
			cout << "3.Up" << endl;
			cout << "4.Down" << endl;
			cout << "5.Facebook" << endl;
			cout << "6.Youtube" << endl;
			cout << "7.News" << endl;			
			cout << "8.Bookmark" << endl;
			cout << "9.History" << endl;
			cout << "10.Read" << endl;
			cout << "11.Stop" << endl;
			cout << "12.Weather" << endl;
			cout << "13.Cricket" << endl;
			cout << "Enter a choice:" << endl;
			cin >> choice1;
			system("cls");


			do_recording(choice1 - 1);

			save_current_recording(choice1 - 1);
			break;
		case 2:
			do_training(default_iteration_value);
			break;
		case 999:
			initialize_iteration_files(default_iteration_value);
			break;

		default:
			exit(0);
			break;
		}

		//save_current_recording(4);
	}
	

	return 0;
}

