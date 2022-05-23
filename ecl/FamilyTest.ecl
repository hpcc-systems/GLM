/*##############################################################################

    HPCC SYSTEMS software Copyright (C) 2022 HPCC Systems®.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
############################################################################## */

#ONWARNING(4531, ignore);

//version tmod='Binomial'
//version tmod='Gamma'
//version tmod='Gaussian'
//version tmod='InvGauss'
//version tmod='Poisson'
//version tmod='Quasibinomial'
//version tmod='Quasipoisson'

// Test that takes the existing BinomialRegression test and parameterizes it to work with all families
// Each version above represents a family and a valid input set

IMPORT ^ AS root;
IMPORT $.^ AS GLMmod;
IMPORT ML_Core.Types AS Types;  
IMPORT ML_Core AS Core;
IMPORT GLMmod.Types AS GLM_Types;
IMPORT GLMmod.Family;
IMPORT GLMmod.Datasets;

Test_Values := RECORD
  STRING8 src;
  REAL8 icept_coef;
  REAL8 sl_coef;
  REAL8 sw_coef;
  REAL8 pl_coef;
  REAL8 pw_coef;
  REAL8 icept_err;
  REAL8 sl_err;
  REAL8 sw_err;
  REAL8 pl_err;
  REAL8 pw_err;
  REAL8 icept_pval;
  REAL8 sl_pval;
  REAL8 sw_pval;
  REAL8 pl_pval;
  REAL8 pw_pval;
  REAL8 aic;
  UNSIGNED4 wi;
END;

IrisData := RECORD
  Types.t_recordID id;
  RECORDOF(Datasets.Iris);
END;

// Takes in a family interface and a set of values and returns any errors
FamilyTest(Family.FamilyInterface family, DATASET(Test_Values) values, INTEGER classRange) := FUNCTION

		IrisData enum_recs(Datasets.Iris rec, UNSIGNED c) := TRANSFORM
			SELF.id := c;
			SELF.class := IF(rec.class=1, 1, classRange);
			SELF := rec;
		END;
		
		iris := PROJECT(Datasets.Iris, enum_recs(LEFT,COUNTER));

		Core.ToField(iris, iris_indep, id, , 1,'sepal_length,sepal_width,petal_length,petal_width');
		Core.ToField(iris, iris_dep, id, , 1, 'class');
		iris_classes := PROJECT(iris_dep, Types.NumericField);

		// Calls functions from the GLM bundle to generate a model
		mdl := GLMmod.GLM(iris_indep, iris_classes, family, DATASET([], Types.NumericField), max_iter:=4, ridge:=0.0).GetModel();
		coef_pval := GLMmod.ExtractBeta_pval(mdl);
		devdet := GLMmod.Deviance_Detail(iris_classes, GLMmod.Predict(coef_pval, iris_indep, family), mdl, family);
		modl_dev := GLMmod.Model_Deviance(devdet, coef_pval);

		// Maximum allowable difference between test values and result values
		REAL8 max_diff := 0.007;

		Compare_Rec := RECORD
			STRING8 src;
			STRING test;
			REAL8 std_value;
			REAL8 tst_value;
			BOOLEAN equal;
		END;

		Compare_Rec check(GLM_Types.pval_Model_Coef p, Test_Values t, UNSIGNED s):=TRANSFORM
			SELF.src := t.src;
			SELF.tst_value := CHOOSE(s, p.w, p.se, p.p_value);
			SELF.std_value := CHOOSE(p.ind_col+1,
						CHOOSE(s, t.icept_coef, t.icept_err, t.icept_pval),
						CHOOSE(s, t.sl_coef, t.sl_err, t.sl_pval),
						CHOOSE(s, t.sw_coef, t.sw_err, t.sw_pval),
						CHOOSE(s, t.pl_coef, t.pl_err, t.pl_pval),
						CHOOSE(s, t.pw_coef, t.pw_err, t.pw_pval));
			SELF.test := CHOOSE(s, 'coef ', 'se ', 'p-val ') + CHOOSE(p.ind_col+1,'Intercept', 'sepal length', 'sepal width', 'petal length', 'petal width');
			SELF.equal := ABS(SELF.tst_value-SELF.std_value) <= max_diff;
		END;

		coef_check := JOIN(coef_pval, values, LEFT.wi=RIGHT.wi, check(LEFT,RIGHT, 1), LOOKUP);
		se_check := JOIN(coef_pval, values, LEFT.wi=RIGHT.wi, check(LEFT,RIGHT, 2), LOOKUP);
		pval_check := JOIN(coef_pval, values, LEFT.wi=RIGHT.wi, check(LEFT, RIGHT, 3), LOOKUP);
											 
		Compare_Rec check_aic(GLM_Types.Deviance_Record d, Test_Values t):=TRANSFORM
			SELF.test := 'AIC';
			SELF.src := t.src;
			SELF.std_value := t.aic;
			SELF.tst_value := d.aic;
			SELF.equal := ABS(SELF.tst_value-SELF.std_value) <= max_diff;
		END;

		aic_check :=  JOIN(modl_dev, values, LEFT.wi=RIGHT.wi, check_aic(LEFT, RIGHT), LOOKUP);
											
		// Concatenates the sets of data
		all_checks := coef_check + se_check + pval_check + aic_check;
		errors := all_checks(NOT equal);
		
		RETURN(errors);
END;

myMod := #IFDEFINED(root.tmod, 'Full');

#IF(myMod = 'Binomial')
	fam := Family.Binomial;
	vals := DATASET([{'sm.logit',
			7.32292705, -0.25274345, -2.77938918,  1.29930595, -2.70427087,
			2.49799533,  0.64945102,  0.78587716,  0.68228159,  1.16265393,
			0.00337306,  0.69715427,  0.00040520,  0.05686405,  0.02002140,
			155.76486509, 1}], Test_Values);
	classRange := 0;
#ELSEIF(myMod = 'Quasibinomial')
	fam := Family.Quasibinomial;
	vals := DATASET([{'sm.logit',
			7.32292705, -0.25274345, -2.77938918,  1.29930595, -2.70427087,
			2.39799533,  0.62345102,  0.75487716,  0.65528159,  1.11265393,
			0.00337306,  0.68515427,  0.00040520,  0.04786405,  0.02002140,
			144.50486509, 1}], Test_Values);
	classRange := 0;
#ELSEIF(myMod = 'Quasipoisson')
	fam := Family.Quasipoisson;
	vals := DATASET([{'sm.logit',
			2.91292705, -0.17274345, -1.41938918,  0.71830595, -1.38427087,
			1.14799533,  0.31945102,  0.33787716,  0.31528159,  0.55565393,
			0.01037306,  0.58515427,  0.00040520,  0.022486405,  0.01202140,
			110.33486509, 1}], Test_Values);
	classRange := 0;
#ELSEIF(myMod = 'Poisson')
	fam := Family.Poisson;
	vals := DATASET([{'sm.logit',
			2.91292705, -0.17274345, -1.41938918,  0.71830595, -1.38427087,
			1.54799533,  0.43345102,  0.45487716,  0.42528159,  0.75265393,
			0.05937306,  0.68515427,  0.00040520,  0.091486405,  0.06502140,
			192.66486509, 1}], Test_Values);
	classRange := 0;
#ELSEIF(myMod = 'Gamma')
	fam := Family.Gamma;
	vals :=	DATASET([{'sm.logit',
			1.01892705, -0.002974345, -0.15938918,  0.077930595, -0.17427087,
			0.15699533,  0.043345102,  0.0443687716,  0.0428159,  0.069265393,
			0.000000000000326007306,  0.9675427,  0.00040520,  0.067086405,  0.01042140,
			192.23486509, 1}], Test_Values);
	classRange := 2;
#ELSEIF(myMod = 'Gaussian')
	fam := Family.Gaussian;
	vals := DATASET([{'sm.logit',
			1.56292705, -0.02274345, -0.44038918,  0.21830595, -0.48427087,
			0.38799533,  0.10845102,  0.11387716,  0.10728159,  0.17865393,
			0.00005937306,  0.84315427,  0.00040520,  0.04286405,  0.006702140,
			165.63486509, 1}], Test_Values);
	classRange := 0;
#ELSEIF(myMod = 'InvGauss')
	fam := Family.InvGauss;
	vals := DATASET([{'sm.logit',
			0.8292705, 0.00164345, -0.177038918,  0.0909830595, -0.213827087,
			0.18899533,  0.054845102,  0.054387716,  0.052728159,  0.086865393,
			0.00005937306,  0.97515427,  0.00040520,  0.0845286405,  0.01326702140,
			300.28486509, 1}], Test_Values);
	classRange := 2;
#END

errors := FamilyTest(fam, vals, classRange);

SEQUENTIAL(OUTPUT('GLM -- FamilyTest', NAMED('TestName')),
	   OUTPUT(IF(EXISTS(errors), 'Fail', 'Pass'), NAMED('Result')),
	   OUTPUT(errors, NAMED('Errors')));	