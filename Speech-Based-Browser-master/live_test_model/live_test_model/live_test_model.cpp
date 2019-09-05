// live_test_model.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "iostream"
#include "fstream"
#include "string"
#include "math.h"
#include "iomanip"
#include "Windows.h"

#define CODE_BOOK_SIZE 32
#define FRAME_SIZE 320
#define FRAMES 130
#define N 5
#define M 32
#define T 130
#define WORDS 11

using namespace std;
#define SCROLL_DOWN_VAL -700
#define SCROLL_UP_VAL 700
#define CONFIRM_BOX_YES 6
#define CONFIRM_BOX_NO 7
#define CONFIRM_BOX_CANCEL 2

static const WORD TAB_SCANCODE = 0x0f;
static const WORD LEFT_ALT_SCANCODE = 0x38;
static const WORD LCTRL_SCANCODE = 0x1d;
static const WORD H_SCANCODE = 0x23;
static const WORD B_SCANCODE = 0x30;
static const WORD J_SCANCODE = 0x24;
static const WORD K_SCANCODE = 0x25;
static const WORD O_SCANCODE = 0x18;
static const WORD P_SCANCODE = 0x19;
static const WORD NUMPAD_5 = 0x4c;
static const WORD LSHIFT_SCANCODE = 0x2a;
static const DWORD keypress_delay_ms = 500;

int read_flag = 0;
int stop_flag = 0;
int close_flag = 0;



void sendkey(WORD scan_code, bool tf_down_up, bool extended = false) {
	INPUT input = { 0 };
	input.type = INPUT_KEYBOARD;
	input.ki.wScan = scan_code;
	input.ki.dwFlags = KEYEVENTF_SCANCODE;
	if (!tf_down_up) input.ki.dwFlags |= KEYEVENTF_KEYUP;
	if (extended)    input.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
	SendInput(1, &input, sizeof(input));
}

void keydown(WORD scan_code, bool extended = false) {
	sendkey(scan_code, true, extended);
}

void keyup(WORD scan_code, bool extended = false) {
	sendkey(scan_code, false, extended);
}

void alt_tab(int tab_count = 1) {
	keydown(LEFT_ALT_SCANCODE);

	Sleep(keypress_delay_ms);
	for (int i = 0; i<tab_count; ++i) {
		keydown(TAB_SCANCODE);
		Sleep(keypress_delay_ms);
		keyup(TAB_SCANCODE);
		Sleep(keypress_delay_ms);
	}
	keyup(LEFT_ALT_SCANCODE);
}

void toggle_to_browser(){
	Sleep(500);
	alt_tab(2);
}

void ctrl_h(){

	keydown(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);
	keydown(H_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(H_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);

}

void open_history(){
	ctrl_h();
}

void ctrl_j(){

	keydown(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);
	keydown(J_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(J_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);

}

void ctrl_k(){

	keydown(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);
	keydown(K_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(K_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);

}

void alt_p(){
	keydown(LEFT_ALT_SCANCODE);
	Sleep(keypress_delay_ms);
	keydown(P_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(P_SCANCODE);
	Sleep(keypress_delay_ms);
	keyup(LEFT_ALT_SCANCODE);
	Sleep(keypress_delay_ms);
}

void start_reading(){
	if (read_flag == 1)
	{
		ctrl_j();
		stop_flag = 1;
	}
}

void stop_reading(){
	printf("callling stop");
	if (stop_flag == 1)
	{
		//printf("\nctrl_k");
		//ctrl_k();
		alt_p();
		stop_flag = 0;
	}
}




void ctrl_shift_b(){

	keydown(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);

	keydown(LSHIFT_SCANCODE);
	Sleep(keypress_delay_ms);

	keydown(O_SCANCODE);
	Sleep(keypress_delay_ms);

	keyup(O_SCANCODE);
	Sleep(keypress_delay_ms);

	keyup(LSHIFT_SCANCODE);
	Sleep(keypress_delay_ms);

	keyup(LCTRL_SCANCODE);
	Sleep(keypress_delay_ms);

}

void open_bookmarks(){
	ctrl_shift_b();
}

void scroll_down(){
	Sleep(500);
	mouse_event(MOUSEEVENTF_WHEEL, 0, 0, SCROLL_DOWN_VAL, 0);
}

void scroll_up(){
	Sleep(500);
	mouse_event(MOUSEEVENTF_WHEEL, 0, 0, SCROLL_UP_VAL, 0);
}

void LeftClick()
{
	printf("\nClick");
	//5scroll_down();
	//mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);

	keydown(NUMPAD_5);
	Sleep(keypress_delay_ms);
	keyup(NUMPAD_5);
	Sleep(keypress_delay_ms);
}

/*int open_confirm_box(char input_msg[]){
const char temp[] = input_msg;
return (MessageBox(nullptr, TEXT(temp) , TEXT("Message"), MB_YESNOCANCEL));
}*/


ifstream in, in1;
ofstream out, out1;

char words[11][100] = { "open", "close", "up", "down", "facebook", "news", "bookmark", "history", "read", "stop", "cricket" };
char* code_book_file = "codebook_32_size.txt";
char observation_sequence_file[100] = "observation_sequence_generated.txt";
char obs_sequence_file[100] = "observation_sequence_live.txt";

int prob_word[11];

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
char* silence_file = "Seminar_silence.txt";
char trimmed_file[100] = "trim_";
char* ri_file = "ri_file.txt";
char* ai_file = "ai_file.txt";
char* ci_file = "ci_file.txt";
char* c_prime_file = "c_prime.txt";
char* universe_file = "universe_digits.txt";
char* hamming_file = "Hamming_window.txt";

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
			//cout << " " << code_book[i][j];
		}
		//cout << endl;
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
	strcpy(normalized_file, "normalized_");
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

	//cout << "Trimmed:" << trimmed_file << endl;
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
	//cout << endl;
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

	//cout << obs[index] << " ";
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
	//	cout << "\nObservation Sequence:" << endl;
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
		//cout << "\n:::::INPUT:" << obs_sequence_file << endl;

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
	cout << "\Word:" << words[iteration] << "\tP(obs/model) : " << Pobs_model;
	cout << endl;
}


void solution_to_problem1(int iteration){
	forward_procedure(iteration);
}

int isSubstring(string s1, string s2)
{
	int m = s1.length();
	int n = s2.length();

	/* A loop to slide pat[] one by one */
	for (int i = 0; i <= n - m; i++) {
		int j;

		/* For current index i, check for pattern match */
		for (j = 0; j < m; j++)
		if (s2[i + j] != s1[j])
			break;

		if (j == m)
			return i;
	}

	return -1;
}




char wnd_title[256];
HWND hwnd;
void operations(int index)
{
	int x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11;
	int i;
	if (system(NULL))
		puts("Proceed");
	else
		exit(1);
	switch (index)
	{
	case 0:
		system("start https://www.google.com/");
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 2) || (i == 3) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}
		close_flag = 1;
		break;
	case 1:
		if (close_flag == 1)
		{
			x1 = MessageBox(nullptr, TEXT("Do you want to close browser?"), TEXT("Message"), MB_YESNOCANCEL);
			if (x1 == CONFIRM_BOX_YES)
			{
				system("taskkill /IM chrome.exe /F");
				close_flag = 0;
				read_flag = 0;
				stop_flag = 0;
				for (i = 0; i < 11; i++)
				{
					if ((i == 1) || (i == 2) || (i == 3) || (i == 6) || (i == 7) || (i == 8) || (i == 9))
						prob_word[i] = 0;
					else
						prob_word[i] = 1;
				}

			}

		}
		break;

	case 2:
		x2 = 5;
		while (x2>0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			//printf("\n%s", wnd_title);

			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x2--;
		}
		scroll_up();
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}

		break;
	case 3:
		x3 = 5;
		while (x3>0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			//printf("\n%s", wnd_title);

			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x3--;
		}
		scroll_down();
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}

		break;

	case 4:

		system("start https://www.facebook.com/");
		close_flag = 1;
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 4) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}
		break;

	case 5:
		system("start https://news.google.com/?hl=en-IN&gl=IN&ceid=IN:en/");
		close_flag = 1;
		read_flag = 1;
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 5) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}
		break;

	case 6:
		x6 = 5;
		while (x6>0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			//printf("\n%s", wnd_title);

			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x6--;
		}
		open_bookmarks();
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 6) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}

		break;

	case 7:
		x7 = 5;
		while (x7 >0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			printf("\n%s", wnd_title);

			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x7--;
		}


		open_history();
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 7) || (i == 8) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}

		break;



	case 8:
		x8 = 5;
		while (x8>0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			//printf("\n%s", wnd_title);

			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x8--;
		}

		start_reading();
		for (i = 0; i < 11; i++)
		{
			if ((i == 1) || (i == 9))
				prob_word[i] = 1;
			else
				prob_word[i] = 0;
		}
		break;



	case 9:
		x9 = 5;
		while (x9>0)
		{

			alt_tab(2);
			Sleep(1000);
			hwnd = GetForegroundWindow(); // get handle of currently active window
			GetWindowTextA(hwnd, wnd_title, sizeof(wnd_title));
			//printf("\n%s", wnd_title);
			if (isSubstring("Google Chrome", wnd_title) != -1)
			{
				break;
			}
			x9--;
		}

		//LeftClick();
		//SetCursor(20,20);
		stop_reading();
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 9))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}
		break;

	case 10:
		system("start https://www.cricbuzz.com/cricket-match/live-scores/");
		close_flag = 1;
		read_flag = 1;
		for (i = 0; i < 11; i++)
		{
			if ((i == 0) || (i == 9) || (i == 10))
				prob_word[i] = 0;
			else
				prob_word[i] = 1;
		}
		break;
	default :
		system("taskkill /IM chrome.exe /F");
		break;


	}

}


void test_word()
{
	int k;
	char ch;
	system("Recording_Module.exe 3 input.wav input.txt");
	cout << "\nRecording successful **Press ENTER to proceed with program**" << endl;
	strcpy(input_file, "input.txt");

	calculate_cepstral_values();
	read_code_book();
	//print_code_book();
	generate_observation_sequence();
	out.open(obs_sequence_file);
	write_observation_sequence();
	out.close();
	output_model_name = 0;
	max_Pobs_model = 0;
	for (k = 0; k < WORDS; k++){
		if (prob_word[k] == 1)

		{
			read_average_model(k);
			solution_to_problem1(k);
		}
	}

	cout << "\nDETECTED WORD:" << words[output_model_name] << endl;
	cout << "\nIs this correct ? c/w" << endl;
	cin >> ch;
	if (ch == 'c')
		operations(output_model_name);
	else if (ch == 'w')
		return;
	else
		printf("\nInvalid");
	//print_code_book();
}

int _tmain(int argc, _TCHAR* argv[])
{
	int ask = 1;
	int k;
	char test_file[100];
	cout << "....... Recording Silence......" << endl;
	system("Recording_Module.exe 3 silence.wav silence_file.txt");
	cout << "\nSilence recorded" << endl;

	for (int i = 0; i <= 10; i++)
	{
		if ((i == 1) || (i == 2) || (i == 3) || (i == 6) || (i == 7) || (i == 8) || (i == 9))
			prob_word[i] = 0;
		else
			prob_word[i] = 1;
	}

	while (ask)
	{
		printf("\nEnter Your choice\n1.SPEAK\n2.Exit\n");
		scanf("%d", &ask);
		switch (ask)
		{
		case 1:
			test_word();
			/*int x ;
			cin >> x;
			operations(x);
			*/
			break;


		case 2: ask = 0;
			exit(0);
			break;
		}
	}
	scanf("%d\n", &ask);
	return 0;

}

