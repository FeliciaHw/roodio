@extends('layouts/signUp')


@section('title', 'ROODIO - Register')


@push('script')
    <script src="{{ asset('js/auth/register.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/logo-with-text.png') }}" alt="logo" class='w-24 drop-shadow-sm drop-shadow-primary-50'>
@endsection


@section('mainTitle', 'SIGN UP')


@section('description', "Let's join for amazing experience!")


@section('content')
    <form action="{{ route('auth.register') }}" method="POST" id='identity'>
        @csrf
        <div class='flex flex-col mb-6'>
            <label for="fullname" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/name.svg') }}" alt="name" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Fullname<span class='text-danger'>*</span></p>
            </label>
            <input type="text" name="fullname" id="fullname" autocomplete="off" placeholder="Ex: John Doe" class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('fullname') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('fullname')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-col mb-6'>
            <label for="email" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/email.svg') }}" alt="email" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Email<span class='text-danger'>*</span></p>
            </label>
            <input type="email" name="email" id="email" inputmode="email" autocomplete="off" placeholder="Ex: john.doe@gmail.com" class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('email') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('email')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-row items-start justify-between mb-6'>
            <div class='flex flex-col'>
                <label for="dob" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/date.svg') }}" alt="email" class='w-6 mr-1'>
                    <p class='text-primary-85 text-body-size'>Date of Birth<span class='text-danger'>*</span></p>
                </label>
                <input datepicker id="default-datepicker" name='dob' type="text" autocomplete="off" placeholder="mm/dd/yyyy" class="not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 w-32 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small lg:w-42 {{ $errors->has('dob') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}">
                <div class="text-error-moderate h-max pt-0.5 text-micro md:text-small">
                    @error('dob')
                        {{ $message }}
                    @enderror
                </div>
            </div>
            <div class='flex flex-col'>
                <label for="gender" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/gender.svg') }}" alt="gender" class='w-6 mr-1'>
                    <p class='text-primary-85'>Gender<span class='text-danger'>*</span></p>
                </label>
                <select name="gender" id="gender" autocomplete="off" class="gender-select w-40 text-small outline-none border-b rounded-md px-1.5 py-1 border-shadedOfGray-50 text-shadedOfGray-60 italic focus:border-secondary-happy-100 focus:border-b-2 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 lg:w-44 {{ $errors->has('gender') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}">
                    <option value="" disabled hidden selected>Gender</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="null">Prefer not to say</option>
                </select>
                <div class="text-error-moderate h-max pt-0.5 text-micro md:text-small">
                    @error('gender')
                        {{ $message }}
                    @enderror
                </div>
            </div>
        </div>
        <div class='flex flex-col mb-8'>
            <label for="country" class='text-body-size flex flex-row mb-2'>
                    <img src="{{ asset('assets/icons/country.svg') }}" alt="country" class='w-6 mr-1'>
                    <p class='text-primary-85'>Country<span class='text-danger'>*</span></p>
            </label>
            <select autocomplete="country" id="country" name="country" autocomplete="off" class="gender-select text-small outline-none border-b rounded-md px-1.5 py-1 border-shadedOfGray-50 text-shadedOfGray-60 italic focus:border-secondary-happy-100 focus:border-b-2 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 {{ $errors->has('country') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}">
                <option value="" disabled hidden selected>Select your country...</option>
                <option value="AF">Afghanistan</option>
                <option value="AX">Åland Islands</option>
                <option value="AL">Albania</option>
                <option value="DZ">Algeria</option>
                <option value="AS">American Samoa</option>
                <option value="AD">Andorra</option>
                <option value="AO">Angola</option>
                <option value="AI">Anguilla</option>
                <option value="AQ">Antarctica</option>
                <option value="AG">Antigua and Barbuda</option>
                <option value="AR">Argentina</option>
                <option value="AM">Armenia</option>
                <option value="AW">Aruba</option>
                <option value="AU">Australia</option>
                <option value="AT">Austria</option>
                <option value="AZ">Azerbaijan</option>
                <option value="BS">Bahamas</option>
                <option value="BH">Bahrain</option>
                <option value="BD">Bangladesh</option>
                <option value="BB">Barbados</option>
                <option value="BY">Belarus</option>
                <option value="BE">Belgium</option>
                <option value="BZ">Belize</option>
                <option value="BJ">Benin</option>
                <option value="BM">Bermuda</option>
                <option value="BT">Bhutan</option>
                <option value="BO">Bolivia (Plurinational State of)</option>
                <option value="BA">Bosnia and Herzegovina</option>
                <option value="BW">Botswana</option>
                <option value="BV">Bouvet Island</option>
                <option value="BR">Brazil</option>
                <option value="IO">British Indian Ocean Territory</option>
                <option value="BN">Brunei Darussalam</option>
                <option value="BG">Bulgaria</option>
                <option value="BF">Burkina Faso</option>
                <option value="BI">Burundi</option>
                <option value="CV">Cabo Verde</option>
                <option value="KH">Cambodia</option>
                <option value="CM">Cameroon</option>
                <option value="CA">Canada</option>
                <option value="BQ">Caribbean Netherlands</option>
                <option value="KY">Cayman Islands</option>
                <option value="CF">Central African Republic</option>
                <option value="TD">Chad</option>
                <option value="CL">Chile</option>
                <option value="CN">China</option>
                <option value="CX">Christmas Island</option>
                <option value="CC">Cocos (Keeling) Islands</option>
                <option value="CO">Colombia</option>
                <option value="KM">Comoros</option>
                <option value="CG">Congo</option>
                <option value="CD">Congo, Democratic Republic of the</option>
                <option value="CK">Cook Islands</option>
                <option value="CR">Costa Rica</option>
                <option value="HR">Croatia</option>
                <option value="CU">Cuba</option>
                <option value="CW">Curaçao</option>
                <option value="CY">Cyprus</option>
                <option value="CZ">Czech Republic</option>
                <option value="CI">Côte d'Ivoire</option>
                <option value="DK">Denmark</option>
                <option value="DJ">Djibouti</option>
                <option value="DM">Dominica</option>
                <option value="DO">Dominican Republic</option>
                <option value="EC">Ecuador</option>
                <option value="EG">Egypt</option>
                <option value="SV">El Salvador</option>
                <option value="GQ">Equatorial Guinea</option>
                <option value="ER">Eritrea</option>
                <option value="EE">Estonia</option>
                <option value="SZ">Eswatini (Swaziland)</option>
                <option value="ET">Ethiopia</option>
                <option value="FK">Falkland Islands (Malvinas)</option>
                <option value="FO">Faroe Islands</option>
                <option value="FJ">Fiji</option>
                <option value="FI">Finland</option>
                <option value="FR">France</option>
                <option value="GF">French Guiana</option>
                <option value="PF">French Polynesia</option>
                <option value="TF">French Southern Territories</option>
                <option value="GA">Gabon</option>
                <option value="GM">Gambia</option>
                <option value="GE">Georgia</option>
                <option value="DE">Germany</option>
                <option value="GH">Ghana</option>
                <option value="GI">Gibraltar</option>
                <option value="GR">Greece</option>
                <option value="GL">Greenland</option>
                <option value="GD">Grenada</option>
                <option value="GP">Guadeloupe</option>
                <option value="GU">Guam</option>
                <option value="GT">Guatemala</option>
                <option value="GG">Guernsey</option>
                <option value="GN">Guinea</option>
                <option value="GW">Guinea-Bissau</option>
                <option value="GY">Guyana</option>
                <option value="HT">Haiti</option>
                <option value="HM">Heard Island and Mcdonald Islands</option>
                <option value="HN">Honduras</option>
                <option value="HK">Hong Kong</option>
                <option value="HU">Hungary</option>
                <option value="IS">Iceland</option>
                <option value="IN">India</option>
                <option value="ID">Indonesia</option>
                <option value="IR">Iran</option>
                <option value="IQ">Iraq</option>
                <option value="IE">Ireland</option>
                <option value="IM">Isle of Man</option>
                <option value="IL">Israel</option>
                <option value="IT">Italy</option>
                <option value="JM">Jamaica</option>
                <option value="JP">Japan</option>
                <option value="JE">Jersey</option>
                <option value="JO">Jordan</option>
                <option value="KZ">Kazakhstan</option>
                <option value="KE">Kenya</option>
                <option value="KI">Kiribati</option>
                <option value="KP">Korea, North</option>
                <option value="KR">Korea, South</option>
                <option value="XK">Kosovo</option>
                <option value="KW">Kuwait</option>
                <option value="KG">Kyrgyzstan</option>
                <option value="LA">Lao People's Democratic Republic</option>
                <option value="LV">Latvia</option>
                <option value="LB">Lebanon</option>
                <option value="LS">Lesotho</option>
                <option value="LR">Liberia</option>
                <option value="LY">Libya</option>
                <option value="LI">Liechtenstein</option>
                <option value="LT">Lithuania</option>
                <option value="LU">Luxembourg</option>
                <option value="MO">Macao</option>
                <option value="MK">Macedonia North</option>
                <option value="MG">Madagascar</option>
                <option value="MW">Malawi</option>
                <option value="MY">Malaysia</option>
                <option value="MV">Maldives</option>
                <option value="ML">Mali</option>
                <option value="MT">Malta</option>
                <option value="MH">Marshall Islands</option>
                <option value="MQ">Martinique</option>
                <option value="MR">Mauritania</option>
                <option value="MU">Mauritius</option>
                <option value="YT">Mayotte</option>
                <option value="MX">Mexico</option>
                <option value="FM">Micronesia</option>
                <option value="MD">Moldova</option>
                <option value="MC">Monaco</option>
                <option value="MN">Mongolia</option>
                <option value="ME">Montenegro</option>
                <option value="MS">Montserrat</option>
                <option value="MA">Morocco</option>
                <option value="MZ">Mozambique</option>
                <option value="MM">Myanmar (Burma)</option>
                <option value="NA">Namibia</option>
                <option value="NR">Nauru</option>
                <option value="NP">Nepal</option>
                <option value="NL">Netherlands</option>
                <option value="AN">Netherlands Antilles</option>
                <option value="NC">New Caledonia</option>
                <option value="NZ">New Zealand</option>
                <option value="NI">Nicaragua</option>
                <option value="NE">Niger</option>
                <option value="NG">Nigeria</option>
                <option value="NU">Niue</option>
                <option value="NF">Norfolk Island</option>
                <option value="MP">Northern Mariana Islands</option>
                <option value="NO">Norway</option>
                <option value="OM">Oman</option>
                <option value="PK">Pakistan</option>
                <option value="PW">Palau</option>
                <option value="PS">Palestine</option>
                <option value="PA">Panama</option>
                <option value="PG">Papua New Guinea</option>
                <option value="PY">Paraguay</option>
                <option value="PE">Peru</option>
                <option value="PH">Philippines</option>
                <option value="PN">Pitcairn Islands</option>
                <option value="PL">Poland</option>
                <option value="PT">Portugal</option>
                <option value="PR">Puerto Rico</option>
                <option value="QA">Qatar</option>
                <option value="RE">Reunion</option>
                <option value="RO">Romania</option>
                <option value="RU">Russian Federation</option>
                <option value="RW">Rwanda</option>
                <option value="BL">Saint Barthelemy</option>
                <option value="SH">Saint Helena</option>
                <option value="KN">Saint Kitts and Nevis</option>
                <option value="LC">Saint Lucia</option>
                <option value="MF">Saint Martin</option>
                <option value="PM">Saint Pierre and Miquelon</option>
                <option value="VC">Saint Vincent and the Grenadines</option>
                <option value="WS">Samoa</option>
                <option value="SM">San Marino</option>
                <option value="ST">Sao Tome and Principe</option>
                <option value="SA">Saudi Arabia</option>
                <option value="SN">Senegal</option>
                <option value="RS">Serbia</option>
                <option value="CS">Serbia and Montenegro</option>
                <option value="SC">Seychelles</option>
                <option value="SL">Sierra Leone</option>
                <option value="SG">Singapore</option>
                <option value="SX">Sint Maarten</option>
                <option value="SK">Slovakia</option>
                <option value="SI">Slovenia</option>
                <option value="SB">Solomon Islands</option>
                <option value="SO">Somalia</option>
                <option value="ZA">South Africa</option>
                <option value="GS">South Georgia and the South Sandwich Islands</option>
                <option value="SS">South Sudan</option>
                <option value="ES">Spain</option>
                <option value="LK">Sri Lanka</option>
                <option value="SD">Sudan</option>
                <option value="SR">Suriname</option>
                <option value="SJ">Svalbard and Jan Mayen</option>
                <option value="SE">Sweden</option>
                <option value="CH">Switzerland</option>
                <option value="SY">Syria</option>
                <option value="TW">Taiwan</option>
                <option value="TJ">Tajikistan</option>
                <option value="TZ">Tanzania</option>
                <option value="TH">Thailand</option>
                <option value="TL">Timor-Leste</option>
                <option value="TG">Togo</option>
                <option value="TK">Tokelau</option>
                <option value="TO">Tonga</option>
                <option value="TT">Trinidad and Tobago</option>
                <option value="TN">Tunisia</option>
                <option value="TR">Turkey (Türkiye)</option>
                <option value="TM">Turkmenistan</option>
                <option value="TC">Turks and Caicos Islands</option>
                <option value="TV">Tuvalu</option>
                <option value="UM">U.S. Outlying Islands</option>
                <option value="UG">Uganda</option>
                <option value="UA">Ukraine</option>
                <option value="AE">United Arab Emirates</option>
                <option value="GB">United Kingdom</option>
                <option value="US">United States</option>
                <option value="UY">Uruguay</option>
                <option value="UZ">Uzbekistan</option>
                <option value="VU">Vanuatu</option>
                <option value="VA">Vatican City Holy See</option>
                <option value="VE">Venezuela</option>
                <option value="VN">Vietnam</option>
                <option value="VG">Virgin Islands, British</option>
                <option value="VI">Virgin Islands, U.S</option>
                <option value="WF">Wallis and Futuna</option>
                <option value="EH">Western Sahara</option>
                <option value="YE">Yemen</option>
                <option value="ZM">Zambia</option>
                <option value="ZW">Zimbabwe</option>
            </select>
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
                @error('country')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div>
            <button type="submit" id='identityBtn' form='identity' class='text-smallBtn font-bold w-full font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150'>Sign Up</button>
            <p class='text-micro text-center md:text-small'>Already have account? <a href="/login" class='font-bold text-secondary-sad-100 hover:text-primary-50'>Let's Login!</a></p>
        </div>
    </form>
@endsection