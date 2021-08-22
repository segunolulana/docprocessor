import unittest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
if __name__ == "__main__":
    from run_sum_helper import DocRetriever
else:
    from dk.py.app.a.smy.doc_retriever import DocRetriever

TEXT1 = """Moved AWS Security Hub score from around 30% to > 80%. Consistently kept Security Hub score up by eliminating 100+ security threats across the entire spectrum of AWS services
Discovered and helped retire even up to 2-year-old unneeded AWS resources
Reduced time to delivery of tasks drastically using Ansible, Boto3, python multiprocessing
Enhanced about 100 S3 buckets in about 4 different parameters
Effected Sig4 signing of AWS S3 requests through Nginx with Lua and with idempotent S3
URL encoding, migrated instances to use IMDSv2, encrypted EBS volumes at rest etc
Senior DevOps Engineer, Paycom (OPay) - July/2018 - March/2020 - Lagos, Nigeria
Opay is a leading mobile money technology company in Nigeria. It is a subsidiary of the browser
company Opera. In 2020, it processed about 80% of bank transfers among mobile operators in Nigeria
Worked on Make, Docker, Jenkins and AWS ECS deployment jobs for Go, MongoDB and Javascript-based projects while adding, approximately, 40285 loc and removing 55182 loc
Supported Jenkins jobs with a total of about 50-100 builds daily and with about 4 builds running concurrently
Increased the reliability of the softwares so they could serve about 100k of customers daily
Improved the success rate of deployment by 50% and also eradicated recurrent intermittent
deployment bugs
"""


class GeneralTest(unittest.TestCase):
    def test_remove_stop_words(self):
        doc_retriever = DocRetriever(None)
        self.assertEqual(doc_retriever.remove_stop_words("PERSONAL"), " *6 ")

    # def test_


if __name__ == "__main__":
    unittest.main()
